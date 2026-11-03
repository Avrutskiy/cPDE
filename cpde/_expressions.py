"""Restricted scalar expression IR. No user text is executed or emitted as code."""
import ast
import re
import math
import numpy as np

# Shared opcodes with cuda/scalar_ops.cuh. Slots a/b hold earlier node indices.
CONSTANT, COMPONENT, FIELD, ADD, SUB, MUL, DIV, POWER, NEG, EXP, LOG, SQRT, SIN, COS, TANH = range(15)
FUNCTIONS = dict(exp=EXP, log=LOG, sqrt=SQRT, sin=SIN, cos=COS, tanh=TANH)


class Expression:
    def __init__(self, text, variables, outputs, parameters=None, freeze=()):
        if not isinstance(text, str): raise ValueError('expression must be a string')
        if set(variables) & set(outputs): raise ValueError('name disjoint outputs and variables')
        self.text, self.variables, self.outputs = text, variables, outputs
        self.parameters = dict(parameters or {})
        for key in self.parameters:
            if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', key) or key in outputs+variables or key in FUNCTIONS or key in ('coord', 'N', 'T'):
                raise ValueError('invalid or reserved parameter name: ' + str(key))
        self.nodes, self.constants, self.fields, self.components = [], [], [], []
        # Parentheses allow ordinary indented, multiline mathematical expressions.
        try: tree = ast.parse('(' + text.strip() + '\n)', mode='eval').body
        except (SyntaxError, RecursionError) as e: raise ValueError('invalid expression') from e
        self._visit(tree)
        self.freeze = {self.canonical(s) for s in freeze}
        if not self.freeze <= set(self.components): raise ValueError('freeze must name referenced network components')
        self.suffixes = [s[2:] if len(s)>1 else '' for s in self.components]
        # COMPONENT b is the local stop-gradient flag.
        for node in self.nodes:
            if node[0] == COMPONENT: node[2] = int(self.components[node[1]] in self.freeze)

    def canonical(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z](?:_[A-Za-z]+)?', name) or name[0] not in self.outputs:
            raise ValueError('unknown network component: ' + str(name))
        suffix = name[2:] if len(name)>1 else ''
        if any(c not in self.variables for c in suffix) or len(suffix)>6:
            raise ValueError('unknown or excessive derivative: ' + name)
        return name[0] + ('_' + ''.join(c*suffix.count(c) for c in self.variables) if suffix else '')

    def _emit(self, op, a=-1, b=-1, value=0):
        if len(self.nodes) >= 512: raise ValueError('expression exceeds 512 IR nodes; split it into costs')
        self.nodes.append([op, a, b]); self.constants.append(float(value))
        return len(self.nodes)-1

    def _visit(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value): raise ValueError('nonfinite literal')
            return self._emit(CONSTANT, value=node.value)
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                if node.id not in self.fields: self.fields.append(node.id)
                return self._emit(FIELD, self.fields.index(node.id))
            symbol = self.canonical(node.id)
            if symbol not in self.components: self.components.append(symbol)
            return self._emit(COMPONENT, self.components.index(symbol))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in ('coord', 'N', 'T') or node.attr not in self.variables or len(node.attr)!=1:
                raise ValueError('only coord.x, N.x and T.x style fields are supported')
            field = node.value.id + '.' + node.attr
            if field not in self.fields: self.fields.append(field)
            return self._emit(FIELD, self.fields.index(field))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a = self._visit(node.operand)
            return self._emit(NEG, a) if isinstance(node.op, ast.USub) else a
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                sign = 1
                if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, (ast.USub, ast.UAdd)):
                    sign = -1 if isinstance(exponent.op, ast.USub) else 1; exponent = exponent.operand
                if not isinstance(exponent, ast.Constant) or type(exponent.value) not in (int, float) or not math.isfinite(exponent.value):
                    raise ValueError('powers require a finite constant exponent')
                return self._emit(POWER, self._visit(node.left), value=sign*exponent.value)
            op = {ast.Add: ADD, ast.Sub: SUB, ast.Mult: MUL, ast.Div: DIV}.get(type(node.op))
            if op is not None: return self._emit(op, self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and len(node.args)==1 and not node.keywords:
            return self._emit(FUNCTIONS[node.func.id], self._visit(node.args[0]))
        raise ValueError('unsupported expression syntax: ' + type(node).__name__)

    def bind(self, inputs, quadrature, count, parameters):
        nodes = np.array(self.nodes, dtype=np.int32)
        if inputs is not None:
            from ._prepare import component
            for i, node in enumerate(nodes):
                if node[0] == COMPONENT: nodes[i, 1] = component(self.components[node[1]], inputs, self.outputs)
        values = []
        for field in self.fields:
            if '.' in field:
                if quadrature is None: raise ValueError('geometry fields require quadrature')
                kind, variable = field.split('.')
                a = quadrature.points if kind == 'coord' else getattr(quadrature, kind)
                if a is None: raise ValueError('quadrature has no ' + kind)
                v = a[:, self.variables.index(variable)]
            else: v = parameters[field]
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 0: v = np.full(count, v)
            if v.shape != (count,) or not np.isfinite(v).all(): raise ValueError('field ' + field + ' must be finite scalar or final-point vector')
            values.append(v)
        return nodes, np.asarray(self.constants), np.asarray(values, dtype=np.float64).reshape(len(values), count)

    def inspect(self):
        return dict(expression=self.text, components=list(self.components), freeze=sorted(self.freeze),
                    fields=list(self.fields), nodes=[list(n) for n in self.nodes], constants=list(self.constants))
