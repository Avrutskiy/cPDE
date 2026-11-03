/*
	q = 1.0f / (1.0f + expf(U[tid]));
	sI = q*(1.0f - q);
	sII = -sI*(1.0f - 2.0f*q);
	sIII = sI*(1 - 6.0f*sI);
	sIV = sII*(1 - 12.0f*sI);
	sV = sI - (5.0f*sII)*(6.0f*sII);
	sVI = sII + 5.0f*(sIV - sII)*(1.0f - 6.0f*sI);
	sVII = (5.0f*sI)*(0.2f - (8.4f*sI)*(3.0f - 4.0f*(5.0f*sI + 5.0f*sIII)));
*/

__device__ float d6p(float eps, float** ADDR, long unsigned int tid) {

	float za, zb, zc, zd, ze, zf, zab, zac, zad, zae, zaf, zbc, zbd,
		zbe, zbf, zcd, zce, zcf, zde, zdf, zef, zabc, zabd, zabe, zabf,
		zacd, zace, zacf, zade, zadf, zaef, zbcd, zbce, zbcf, zbde, zbdf,
		zbef, zcde, zcdf, zcef, zdef, zabcd, zabce, zabcf, zabde, zabdf, zabef, 
		zacde, zacdf, zacef, zadef, zbcde, zbcdf, zbcef, zbdef, zcdef, zabcde, zabcdf,
		zabcef, zabdef, zacdef, zbcdef, zabcdef;

	float q, s0, sI, sII, sIII, sIV, sV, sVI;

	float M1, M2, M3, M4, M5, M6;

	za = *(ADDR[1] + tid);
	zb = *(ADDR[2] + tid);
	zc = *(ADDR[3] + tid);
	zd = *(ADDR[4] + tid);
	ze = *(ADDR[5] + tid);
	zf = *(ADDR[6] + tid);
	zab = *(ADDR[7] + tid);
	zac = *(ADDR[8] + tid);
	zad = *(ADDR[9] + tid);
	zae = *(ADDR[10] + tid);
	zaf = *(ADDR[11] + tid);
	zbc = *(ADDR[12] + tid);
	zbd = *(ADDR[13] + tid);
	zbe = *(ADDR[14] + tid);
	zbf = *(ADDR[15] + tid);
	zcd = *(ADDR[16] + tid);
	zce = *(ADDR[17] + tid);
	zcf = *(ADDR[18] + tid);
	zde = *(ADDR[19] + tid);
	zdf = *(ADDR[20] + tid);
	zef = *(ADDR[21] + tid);
	zabc = *(ADDR[22] + tid);
	zabd = *(ADDR[23] + tid);
	zabe = *(ADDR[24] + tid);
	zabf = *(ADDR[25] + tid);
	zacd = *(ADDR[26] + tid);
	zace = *(ADDR[27] + tid);
	zacf = *(ADDR[28] + tid);
	zade = *(ADDR[29] + tid);
	zadf = *(ADDR[30] + tid);
	zaef = *(ADDR[31] + tid);
	zbcd = *(ADDR[32] + tid);
	zbce = *(ADDR[33] + tid);
	zbcf = *(ADDR[34] + tid);
	zbde = *(ADDR[35] + tid);
	zbdf = *(ADDR[36] + tid);
	zbef = *(ADDR[37] + tid);
	zcde = *(ADDR[38] + tid);
	zcdf = *(ADDR[39] + tid);
	zcef = *(ADDR[40] + tid);
	zdef = *(ADDR[41] + tid);
	zabcd = *(ADDR[42] + tid);
	zabce = *(ADDR[43] + tid);
	zabcf = *(ADDR[44] + tid);
	zabde = *(ADDR[45] + tid);
	zabdf = *(ADDR[46] + tid);
	zabef = *(ADDR[47] + tid);
	zacde = *(ADDR[48] + tid);
	zacdf = *(ADDR[49] + tid);
	zacef = *(ADDR[50] + tid);
	zadef = *(ADDR[51] + tid);
	zbcde = *(ADDR[52] + tid);
	zbcdf = *(ADDR[53] + tid);
	zbcef = *(ADDR[54] + tid);
	zbdef = *(ADDR[55] + tid);
	zcdef = *(ADDR[56] + tid);
	zabcde = *(ADDR[57] + tid);
	zabcdf = *(ADDR[58] + tid);
	zabcef = *(ADDR[59] + tid);
	zabdef = *(ADDR[60] + tid);
	zacdef = *(ADDR[61] + tid);
	zbcdef = *(ADDR[62] + tid);
	zabcdef = *(ADDR[63] + tid);

	q = eps;
	s0 = q * (1.0f - q);
	sI = -s0 * (1.0f - 2.0f*q);
	sII = s0 * (1 - 6.0f*s0);
	sIII = sI * (1 - 12.0f*s0);
	sIV = s0 - (5.0f*sI)*(6.0f*sI);
	sV = sI + 5.0f*(sIII - sI)*(1.0f - 6.0f*s0);
	sVI = (5.0f*s0)*(0.2f - (8.4f*s0)*(3.0f - 4.0f*(5.0f*s0 + 5.0f*sII)));

	M1 = zabcdef;

	M2 = zacdef * zb + zadef * zbc + zaef * zbcd;
	M2 += zaf * zbcde + za * zbcdef + zae * zbcdf;
	M2 += zadf * zbce + zad * zbcef + zade * zbcf;
	M2 += zacef * zbd + zacf * zbde + zac * zbdef;
	M2 += zace * zbdf + zacdf * zbe + zacd * zbef;
	M2 += zacde * zbf + zabdef * zc + zabef * zcd;
	M2 += zabf * zcde + zab * zcdef + zabe * zcdf;
	M2 += zabdf * zce + zabd * zcef + zabde * zcf;
	M2 += zabcef * zd + zabcf * zde + zabc * zdef;
	M2 += zabce * zdf + zabcdf * ze + zabcd * zef;
	M2 += zabcde * zf;


	M3 = zadef * zb*zc + zaef * zbd*zc + zaf * zbde*zc;
	M3 += za * zbdef*zc + zae * zbdf*zc + zadf * zbe*zc;
	M3 += zad * zbef*zc + zade * zbf*zc + zaef * zb*zcd;
	M3 += zaf * zbe*zcd + za * zbef*zcd + zae * zbf*zcd;
	M3 += zaf * zb*zcde + za * zbf*zcde + za * zb*zcdef;
	M3 += zae * zb*zcdf + za * zbe*zcdf + zadf * zb*zce;
	M3 += zaf * zbd*zce + za * zbdf*zce + zad * zbf*zce;
	M3 += zad * zb*zcef + za * zbd*zcef + zade * zb*zcf;
	M3 += zae * zbd*zcf + za * zbde*zcf + zad * zbe*zcf;
	M3 += zacef * zb*zd + zaef * zbc*zd + zaf * zbce*zd;
	M3 += za * zbcef*zd + zae * zbcf*zd + zacf * zbe*zd;
	M3 += zac * zbef*zd + zace * zbf*zd + zabef * zc*zd;
	M3 += zabf * zce*zd + zab * zcef*zd + zabe * zcf*zd;
	M3 += zacf * zb*zde + zaf * zbc*zde + za * zbcf*zde;
	M3 += zac * zbf*zde + zabf * zc*zde + zab * zcf*zde;
	M3 += zac * zb*zdef + za * zbc*zdef + zab * zc*zdef;
	M3 += zace * zb*zdf + zae * zbc*zdf + za * zbce*zdf;
	M3 += zac * zbe*zdf + zabe * zc*zdf + zab * zce*zdf;
	M3 += zacdf * zb*ze + zadf * zbc*ze + zaf * zbcd*ze;
	M3 += za * zbcdf*ze + zad * zbcf*ze + zacf * zbd*ze;
	M3 += zac * zbdf*ze + zacd * zbf*ze + zabdf * zc*ze;
	M3 += zabf * zcd*ze + zab * zcdf*ze + zabd * zcf*ze;
	M3 += zabcf * zd*ze + zabc * zdf*ze + zacd * zb*zef;
	M3 += zad * zbc*zef + za * zbcd*zef + zac * zbd*zef;
	M3 += zabd * zc*zef + zab * zcd*zef + zabc * zd*zef;
	M3 += zacde * zb*zf + zade * zbc*zf + zae * zbcd*zf;
	M3 += za * zbcde*zf + zad * zbce*zf + zace * zbd*zf;
	M3 += zac * zbde*zf + zacd * zbe*zf + zabde * zc*zf;
	M3 += zabe * zcd*zf + zab * zcde*zf + zabd * zce*zf;
	M3 += zabce * zd*zf + zabc * zde*zf + zabcd * ze*zf;


	M4 = zaef * zb*zc*zd + zaf * zbe*zc*zd + za * zbef*zc*zd;
	M4 += zae * zbf*zc*zd + zaf * zb*zce*zd + za * zbf*zce*zd;
	M4 += za * zb*zcef*zd + zae * zb*zcf*zd + za * zbe*zcf*zd;
	M4 += zaf * zb*zc*zde + za * zbf*zc*zde + za * zb*zcf*zde;
	M4 += za * zb*zc*zdef + zae * zb*zc*zdf + za * zbe*zc*zdf;
	M4 += za * zb*zce*zdf + zadf * zb*zc*ze + zaf * zbd*zc*ze;
	M4 += za * zbdf*zc*ze + zad * zbf*zc*ze + zaf * zb*zcd*ze;
	M4 += za * zbf*zcd*ze + za * zb*zcdf*ze + zad * zb*zcf*ze;
	M4 += za * zbd*zcf*ze + zacf * zb*zd*ze + zaf * zbc*zd*ze;
	M4 += za * zbcf*zd*ze + zac * zbf*zd*ze + zabf * zc*zd*ze;
	M4 += zab * zcf*zd*ze + zac * zb*zdf*ze + za * zbc*zdf*ze;
	M4 += zab * zc*zdf*ze + zad * zb*zc*zef + za * zbd*zc*zef;
	M4 += za * zb*zcd*zef + zac * zb*zd*zef + za * zbc*zd*zef;
	M4 += zab * zc*zd*zef + zade * zb*zc*zf + zae * zbd*zc*zf;
	M4 += za * zbde*zc*zf + zad * zbe*zc*zf + zae * zb*zcd*zf;
	M4 += za * zbe*zcd*zf + za * zb*zcde*zf + zad * zb*zce*zf;
	M4 += za * zbd*zce*zf + zace * zb*zd*zf + zae * zbc*zd*zf;
	M4 += za * zbce*zd*zf + zac * zbe*zd*zf + zabe * zc*zd*zf;
	M4 += zab * zce*zd*zf + zac * zb*zde*zf + za * zbc*zde*zf;
	M4 += zab * zc*zde*zf + zacd * zb*ze*zf + zad * zbc*ze*zf;
	M4 += za * zbcd*ze*zf + zac * zbd*ze*zf + zabd * zc*ze*zf;
	M4 += zab * zcd*ze*zf + zabc * zd*ze*zf;


	M5 = zaf * zb*zc*zd*ze + za * zbf*zc*zd*ze + za * zb*zcf*zd*ze;
	M5 += za * zb*zc*zdf*ze + za * zb*zc*zd*zef + zae * zb*zc*zd*zf;
	M5 += za * zbe*zc*zd*zf + za * zb*zce*zd*zf + za * zb*zc*zde*zf;
	M5 += zad * zb*zc*ze*zf + za * zbd*zc*ze*zf + za * zb*zcd*ze*zf;
	M5 += zac * zb*zd*ze*zf + za * zbc*zd*ze*zf + zab * zc*zd*ze*zf;

	M6 = za*zb*zc*zd*ze*zf;

	return (sI*M1 + sII * M2 + sIII * M3 + sIV * M4 + sV * M5 + sVI * M6);

}

__device__ float d5p(float eps, float** ADDR, long unsigned int tid){
	float za, zb, zc, zd, ze, zab, zac, zad, zae, zbc, zbd, zbe, zcd, zce, zde, zabc, zabd, zabe, zacd, zace, zade, zbcd, zbce, zbde, zcde, zabcd, zabce, zabde, zacde, zbcde, zabcde;
	float q, s0, sI, sII, sIII, sIV, sV;
	float m1, m2, m3, m4;
	

		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zd = *(ADDR[4] + tid);
		ze = *(ADDR[5] + tid);
		zab = *(ADDR[6] + tid);
		zac = *(ADDR[7] + tid);
		zad = *(ADDR[8] + tid);
		zae = *(ADDR[9] + tid);
		zbc = *(ADDR[10] + tid);
		zbd = *(ADDR[11] + tid);
		zbe = *(ADDR[12] + tid);
		zcd = *(ADDR[13] + tid);
		zce = *(ADDR[14] + tid);
		zde = *(ADDR[15] + tid);
		zabc = *(ADDR[16] + tid);
		zabd = *(ADDR[17] + tid);
		zabe = *(ADDR[18] + tid);
		zacd = *(ADDR[19] + tid);
		zace = *(ADDR[20] + tid);
		zade = *(ADDR[21] + tid);
		zbcd = *(ADDR[22] + tid);
		zbce = *(ADDR[23] + tid);
		zbde = *(ADDR[24] + tid);
		zcde = *(ADDR[25] + tid);
		zabcd = *(ADDR[26] + tid);
		zabce = *(ADDR[27] + tid);
		zabde = *(ADDR[28] + tid);
		zacde = *(ADDR[29] + tid);
		zbcde = *(ADDR[30] + tid);
		zabcde = *(ADDR[31] + tid);

		q = eps;
		s0 = q*(1.0f - q);
		sI = -s0*(1.0f - 2.0f*q);
		sII = s0*(1 - 6.0f*s0);
		sIII = sI*(1 - 12.0f*s0);
		sIV = s0 - (5.0f*sI)*(6.0f*sI);
		sV = sI + 5.0f*(sIII - sI)*(1.0f - 6.0f*s0);

		m1 = zacde * zb + zade * zbc + zae * zbcd + za * zbcde;
		m1 += zad * zbce + zace * zbd + zac * zbde + zacd * zbe;
		m1 += zabde * zc + zabe * zcd + zab * zcde + zabd * zce;
		m1 += zabce * zd + zabc * zde + zabcd * ze;

		m2 = zade * zb*zc + zae * zbd*zc + za * zbde*zc + zad * zbe*zc;
		m2 += zae * zb*zcd + za * zbe*zcd + za * zb*zcde + zad * zb*zce;
		m2 += za * zbd*zce + zace * zb*zd + zae * zbc*zd + za * zbce*zd;
		m2 += zac * zbe*zd + zabe * zc*zd + zab * zce*zd + zac * zb*zde;
		m2 += za * zbc*zde + zab * zc*zde + zacd * zb*ze + zad * zbc*ze;
		m2 += za * zbcd*ze + zac * zbd*ze + zabd * zc*ze + zab * zcd*ze + zabc * zd*ze;

		m3 = zae * zb*zc*zd + za * zbe*zc*zd + za * zb*zce*zd + za * zb*zc*zde;
		m3 += zad * zb*zc*ze + za * zbd*zc*ze + za * zb*zcd*ze;
		m3 += zac * zb*zd*ze + za * zbc*zd*ze + zab * zc*zd*ze;

		m4 = sI * zabcde + sII * m1 + sIII * m2 + sIV * m3 + sV * za*zb*zc*zd*ze;

		return m4;

}

__device__ float d4p(float eps, float** ADDR, long unsigned int tid){
	float za, zb, zc, zd, zab, zac, zad, zbc, zbd, zcd, zabc, zabd, zacd, zbcd, zabcd;
	float q, s0, sI, sII, sIII, sIV;

		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zd = *(ADDR[4] + tid);
		zab = *(ADDR[5] + tid);
		zac = *(ADDR[6] + tid);
		zad = *(ADDR[7] + tid);
		zbc = *(ADDR[8] + tid);
		zbd = *(ADDR[9] + tid);
		zcd = *(ADDR[10] + tid);
		zabc = *(ADDR[11] + tid);
		zabd = *(ADDR[12] + tid);
		zacd = *(ADDR[13] + tid);
		zbcd = *(ADDR[14] + tid);
		zabcd = *(ADDR[15] + tid);

		q = eps;
		s0 = q*(1.0f - q);
		sI = -s0*(1.0f - 2.0f*q);
		sII = s0*(1 - 6.0f*s0);
		sIII = sI*(1 - 12.0f*s0);
		sIV = s0 - (5.0f*sI)*(6.0f*sI);

			return
			(sI*zabcd +
			sII*(zacd*zb + zad*zbc + za*zbcd + zac*zbd + zabd*zc + zab*zcd + zabc*zd) +
			sIII*(zad*zb*zc + za*zbd*zc + za*zb*zcd + zac*zb*zd + za*zbc*zd + zab*zc*zd) +
			sIV*za*zb*zc*zd);

}

__device__ float d3p(float eps, float** ADDR, long unsigned int tid){
	float za, zb, zc, zab, zac, zbc, zabc;
	float q, s0, sI, sII, sIII;

		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zab = *(ADDR[4] + tid);
		zac = *(ADDR[5] + tid);
		zbc = *(ADDR[6] + tid);
		zabc = *(ADDR[7] + tid);

		q = eps;
		s0 = q*(1.0f - q);
		sI = -s0*(1.0f - 2.0f*q);
		sII = s0*(1 - 6.0f*s0);
		sIII = sI*(1 - 12.0f*s0);

		return (sI*zabc + sII*(zac*zb + za*zbc + zab*zc) + sIII*za*zb*zc);

}

__device__ float d2p(float eps, float** ADDR, long unsigned int tid){

	float za, zb, zab;
	float q, s0, sI, sII;

		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zab = *(ADDR[3] + tid);

		q = eps;
		s0 = q*(1.0f - q);
		sI = -s0*(1.0f - 2.0f*q);
		sII = s0*(1 - 6.0f*s0);

		return (sI*zab + sII*za*zb);

}

__device__ float d1p(float eps, float** ADDR, long unsigned int tid){

	float za;
	float q, s0, sI;

		za = *(ADDR[1] + tid);

		q = eps;
		s0 = q*(1.0f - q);
		sI = -s0*(1.0f - 2.0f*q);

		return sI*za;

}

__device__ float DPlus(float eps, int order, float** STCK, long unsigned int tid){
	if (order == 1) return d1p(eps, STCK, tid);
	if (order == 2) return d2p(eps, STCK, tid);
	if (order == 3) return d3p(eps, STCK, tid);
	if (order == 4) return d4p(eps, STCK, tid);
	if (order == 5) return d5p(eps, STCK, tid);
	if (order == 6) return d6p(eps, STCK, tid);
	return 0.0f;
}

__global__ void back_step(int* S, int Sh, float** ADDR, int* dTab, int* Crs, float** STCK, int H, int W){

	int i;
	float* TS;
	float z;
	float eps;
	float sI;
	int j;
	int dn;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){
		i = 0;

		z = *(STCK[0] + tid);
		eps = 1.0f / (1.0f + expf(z));
		sI = eps*(1.0f - eps);

		while (i < Sh) {
				TS = ADDR[i] + tid;
				(*TS) *= sI;
				j = i + 1;
				while (j < Sh) {
					dn = dTab[i*Sh + j];
					if (dn != -1) (*TS) += ((float)Crs[i*Sh + j]) * (*(ADDR[j] + tid))*DPlus(eps, S[dn], STCK + 64 * dn, tid);
					j++;
				}
				i++;
		}
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d6(float* dest, float** ADDR, int H, int W) {

	float z, za, zb, zc, zd, ze, zf, zab, zac, zad, zae, zaf, zbc, zbd, zbe, zbf, zcd, zce, zcf, zde,
		zdf, zef, zabc, zabd, zabe, zabf, zacd, zace, zacf, zade, zadf, zaef, zbcd, zbce, zbcf, zbde, zbdf,
		zbef, zcde, zcdf, zcef, zdef, zabcd, zabce, zabcf, zabde, zabdf, zabef, zacde, zacdf, zacef, zadef,
		zbcde, zbcdf, zbcef, zbdef, zcdef, zabcde, zabcdf, zabcef, zabdef, zacdef, zbcdef, zabcdef;

	float M1, M2, M3, M4, M5, M6;

	float q, sI, sII, sIII, sIV, sV, sVI;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < H*W) {

		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zd = *(ADDR[4] + tid);
		ze = *(ADDR[5] + tid);
		zf = *(ADDR[6] + tid);
		zab = *(ADDR[7] + tid);
		zac = *(ADDR[8] + tid);
		zad = *(ADDR[9] + tid);
		zae = *(ADDR[10] + tid);
		zaf = *(ADDR[11] + tid);
		zbc = *(ADDR[12] + tid);
		zbd = *(ADDR[13] + tid);
		zbe = *(ADDR[14] + tid);
		zbf = *(ADDR[15] + tid);
		zcd = *(ADDR[16] + tid);
		zce = *(ADDR[17] + tid);
		zcf = *(ADDR[18] + tid);
		zde = *(ADDR[19] + tid);
		zdf = *(ADDR[20] + tid);
		zef = *(ADDR[21] + tid);
		zabc = *(ADDR[22] + tid);
		zabd = *(ADDR[23] + tid);
		zabe = *(ADDR[24] + tid);
		zabf = *(ADDR[25] + tid);
		zacd = *(ADDR[26] + tid);
		zace = *(ADDR[27] + tid);
		zacf = *(ADDR[28] + tid);
		zade = *(ADDR[29] + tid);
		zadf = *(ADDR[30] + tid);
		zaef = *(ADDR[31] + tid);
		zbcd = *(ADDR[32] + tid);
		zbce = *(ADDR[33] + tid);
		zbcf = *(ADDR[34] + tid);
		zbde = *(ADDR[35] + tid);
		zbdf = *(ADDR[36] + tid);
		zbef = *(ADDR[37] + tid);
		zcde = *(ADDR[38] + tid);
		zcdf = *(ADDR[39] + tid);
		zcef = *(ADDR[40] + tid);
		zdef = *(ADDR[41] + tid);
		zabcd = *(ADDR[42] + tid);
		zabce = *(ADDR[43] + tid);
		zabcf = *(ADDR[44] + tid);
		zabde = *(ADDR[45] + tid);
		zabdf = *(ADDR[46] + tid);
		zabef = *(ADDR[47] + tid);
		zacde = *(ADDR[48] + tid);
		zacdf = *(ADDR[49] + tid);
		zacef = *(ADDR[50] + tid);
		zadef = *(ADDR[51] + tid);
		zbcde = *(ADDR[52] + tid);
		zbcdf = *(ADDR[53] + tid);
		zbcef = *(ADDR[54] + tid);
		zbdef = *(ADDR[55] + tid);
		zcdef = *(ADDR[56] + tid);
		zabcde = *(ADDR[57] + tid);
		zabcdf = *(ADDR[58] + tid);
		zabcef = *(ADDR[59] + tid);
		zabdef = *(ADDR[60] + tid);
		zacdef = *(ADDR[61] + tid);
		zbcdef = *(ADDR[62] + tid);
		zabcdef = *(ADDR[63] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q * (1.0f - q);
		sII = -sI * (1.0f - 2.0f*q);
		sIII = sI * (1.0f - 6.0f*sI);
		sIV = sII * (1.0f - 12.0f*sI);
		sV = sI - (5.0f*sII)*(6.0f*sII);
		sVI = sII + 5.0f*(sIV - sII)*(1.0f - 6.0f*sI);

		M1 = zabcdef;

		M2 = zacdef * zb + zadef * zbc + zaef * zbcd;
		M2 += zaf * zbcde + za * zbcdef + zae * zbcdf;
		M2 += zadf * zbce + zad * zbcef + zade * zbcf;
		M2 += zacef * zbd + zacf * zbde + zac * zbdef;
		M2 += zace * zbdf + zacdf * zbe + zacd * zbef;
		M2 += zacde * zbf + zabdef * zc + zabef * zcd;
		M2 += zabf * zcde + zab * zcdef + zabe * zcdf;
		M2 += zabdf * zce + zabd * zcef + zabde * zcf;
		M2 += zabcef * zd + zabcf * zde + zabc * zdef;
		M2 += zabce * zdf + zabcdf * ze + zabcd * zef;
		M2 += zabcde * zf;


		M3 = zadef * zb*zc + zaef * zbd*zc + zaf * zbde*zc;
		M3 += za * zbdef*zc + zae * zbdf*zc + zadf * zbe*zc;
		M3 += zad * zbef*zc + zade * zbf*zc + zaef * zb*zcd;
		M3 += zaf * zbe*zcd + za * zbef*zcd + zae * zbf*zcd;
		M3 += zaf * zb*zcde + za * zbf*zcde + za * zb*zcdef;
		M3 += zae * zb*zcdf + za * zbe*zcdf + zadf * zb*zce;
		M3 += zaf * zbd*zce + za * zbdf*zce + zad * zbf*zce;
		M3 += zad * zb*zcef + za * zbd*zcef + zade * zb*zcf;
		M3 += zae * zbd*zcf + za * zbde*zcf + zad * zbe*zcf;
		M3 += zacef * zb*zd + zaef * zbc*zd + zaf * zbce*zd;
		M3 += za * zbcef*zd + zae * zbcf*zd + zacf * zbe*zd;
		M3 += zac * zbef*zd + zace * zbf*zd + zabef * zc*zd;
		M3 += zabf * zce*zd + zab * zcef*zd + zabe * zcf*zd;
		M3 += zacf * zb*zde + zaf * zbc*zde + za * zbcf*zde;
		M3 += zac * zbf*zde + zabf * zc*zde + zab * zcf*zde;
		M3 += zac * zb*zdef + za * zbc*zdef + zab * zc*zdef;
		M3 += zace * zb*zdf + zae * zbc*zdf + za * zbce*zdf;
		M3 += zac * zbe*zdf + zabe * zc*zdf + zab * zce*zdf;
		M3 += zacdf * zb*ze + zadf * zbc*ze + zaf * zbcd*ze;
		M3 += za * zbcdf*ze + zad * zbcf*ze + zacf * zbd*ze;
		M3 += zac * zbdf*ze + zacd * zbf*ze + zabdf * zc*ze;
		M3 += zabf * zcd*ze + zab * zcdf*ze + zabd * zcf*ze;
		M3 += zabcf * zd*ze + zabc * zdf*ze + zacd * zb*zef;
		M3 += zad * zbc*zef + za * zbcd*zef + zac * zbd*zef;
		M3 += zabd * zc*zef + zab * zcd*zef + zabc * zd*zef;
		M3 += zacde * zb*zf + zade * zbc*zf + zae * zbcd*zf;
		M3 += za * zbcde*zf + zad * zbce*zf + zace * zbd*zf;
		M3 += zac * zbde*zf + zacd * zbe*zf + zabde * zc*zf;
		M3 += zabe * zcd*zf + zab * zcde*zf + zabd * zce*zf;
		M3 += zabce * zd*zf + zabc * zde*zf + zabcd * ze*zf;


		M4 = zaef * zb*zc*zd + zaf * zbe*zc*zd + za * zbef*zc*zd;
		M4 += zae * zbf*zc*zd + zaf * zb*zce*zd + za * zbf*zce*zd;
		M4 += za * zb*zcef*zd + zae * zb*zcf*zd + za * zbe*zcf*zd;
		M4 += zaf * zb*zc*zde + za * zbf*zc*zde + za * zb*zcf*zde;
		M4 += za * zb*zc*zdef + zae * zb*zc*zdf + za * zbe*zc*zdf;
		M4 += za * zb*zce*zdf + zadf * zb*zc*ze + zaf * zbd*zc*ze;
		M4 += za * zbdf*zc*ze + zad * zbf*zc*ze + zaf * zb*zcd*ze;
		M4 += za * zbf*zcd*ze + za * zb*zcdf*ze + zad * zb*zcf*ze;
		M4 += za * zbd*zcf*ze + zacf * zb*zd*ze + zaf * zbc*zd*ze;
		M4 += za * zbcf*zd*ze + zac * zbf*zd*ze + zabf * zc*zd*ze;
		M4 += zab * zcf*zd*ze + zac * zb*zdf*ze + za * zbc*zdf*ze;
		M4 += zab * zc*zdf*ze + zad * zb*zc*zef + za * zbd*zc*zef;
		M4 += za * zb*zcd*zef + zac * zb*zd*zef + za * zbc*zd*zef;
		M4 += zab * zc*zd*zef + zade * zb*zc*zf + zae * zbd*zc*zf;
		M4 += za * zbde*zc*zf + zad * zbe*zc*zf + zae * zb*zcd*zf;
		M4 += za * zbe*zcd*zf + za * zb*zcde*zf + zad * zb*zce*zf;
		M4 += za * zbd*zce*zf + zace * zb*zd*zf + zae * zbc*zd*zf;
		M4 += za * zbce*zd*zf + zac * zbe*zd*zf + zabe * zc*zd*zf;
		M4 += zab * zce*zd*zf + zac * zb*zde*zf + za * zbc*zde*zf;
		M4 += zab * zc*zde*zf + zacd * zb*ze*zf + zad * zbc*ze*zf;
		M4 += za * zbcd*ze*zf + zac * zbd*ze*zf + zabd * zc*ze*zf;
		M4 += zab * zcd*ze*zf + zabc * zd*ze*zf;


		M5 = zaf * zb*zc*zd*ze + za * zbf*zc*zd*ze + za * zb*zcf*zd*ze;
		M5 += za * zb*zc*zdf*ze + za * zb*zc*zd*zef + zae * zb*zc*zd*zf;
		M5 += za * zbe*zc*zd*zf + za * zb*zce*zd*zf + za * zb*zc*zde*zf;
		M5 += zad * zb*zc*ze*zf + za * zbd*zc*ze*zf + za * zb*zcd*ze*zf;
		M5 += zac * zb*zd*ze*zf + za * zbc*zd*ze*zf + zab * zc*zd*ze*zf;

		M6 = za*zb*zc*zd*ze*zf;

		dest[tid] = (sI*M1 + sII * M2 + sIII * M3 + sIV * M4 + sV * M5 + sVI * M6);

		tid += blockDim.x * gridDim.x;
	}

}

__global__ void d5(float* dest, float** ADDR, int H, int W){
	float z, za, zb, zc, zd, ze, zab, zac, zad, zae, zbc, zbd, zbe, zcd, zce, zde, zabc, zabd, zabe, zacd, zace, zade, zbcd, zbce, zbde, zcde, zabcd, zabce, zabde, zacde, zbcde, zabcde;
	float q, sI, sII, sIII, sIV, sV;
	float m1, m2, m3, m4;
	

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){

		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zd = *(ADDR[4] + tid);
		ze = *(ADDR[5] + tid);
		zab = *(ADDR[6] + tid);
		zac = *(ADDR[7] + tid);
		zad = *(ADDR[8] + tid);
		zae = *(ADDR[9] + tid);
		zbc = *(ADDR[10] + tid);
		zbd = *(ADDR[11] + tid);
		zbe = *(ADDR[12] + tid);
		zcd = *(ADDR[13] + tid);
		zce = *(ADDR[14] + tid);
		zde = *(ADDR[15] + tid);
		zabc = *(ADDR[16] + tid);
		zabd = *(ADDR[17] + tid);
		zabe = *(ADDR[18] + tid);
		zacd = *(ADDR[19] + tid);
		zace = *(ADDR[20] + tid);
		zade = *(ADDR[21] + tid);
		zbcd = *(ADDR[22] + tid);
		zbce = *(ADDR[23] + tid);
		zbde = *(ADDR[24] + tid);
		zcde = *(ADDR[25] + tid);
		zabcd = *(ADDR[26] + tid);
		zabce = *(ADDR[27] + tid);
		zabde = *(ADDR[28] + tid);
		zacde = *(ADDR[29] + tid);
		zbcde = *(ADDR[30] + tid);
		zabcde = *(ADDR[31] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q*(1.0f - q);
		sII = -sI*(1.0f - 2.0f*q);
		sIII = sI*(1.0f - 6.0f*sI);
		sIV = sII*(1.0f - 12.0f*sI);
		sV = sI - (5.0f*sII)*(6.0f*sII);

		m1 = zacde * zb + zade * zbc + zae * zbcd + za * zbcde;
		m1 += zad * zbce + zace * zbd + zac * zbde + zacd * zbe;
		m1 += zabde * zc + zabe * zcd + zab * zcde + zabd * zce;
		m1 += zabce * zd + zabc * zde + zabcd * ze;

		m2 = zade * zb*zc + zae * zbd*zc + za * zbde*zc + zad * zbe*zc;
		m2 += zae * zb*zcd + za * zbe*zcd + za * zb*zcde + zad * zb*zce;
		m2 += za * zbd*zce + zace * zb*zd + zae * zbc*zd + za * zbce*zd;
		m2 += zac * zbe*zd + zabe * zc*zd + zab * zce*zd + zac * zb*zde;
		m2 += za * zbc*zde + zab * zc*zde + zacd * zb*ze + zad * zbc*ze;
		m2 += za * zbcd*ze + zac * zbd*ze + zabd * zc*ze + zab * zcd*ze + zabc * zd*ze;

		m3 = zae * zb*zc*zd + za * zbe*zc*zd + za * zb*zce*zd + za * zb*zc*zde;
		m3 += zad * zb*zc*ze + za * zbd*zc*ze + za * zb*zcd*ze;
		m3 += zac * zb*zd*ze + za * zbc*zd*ze + zab * zc*zd*ze;

		m4 = sI * zabcde + sII * m1 + sIII * m2 + sIV * m3 + sV * za*zb*zc*zd*ze;

		dest[tid] = m4;

		tid += blockDim.x * gridDim.x;
	}
	

}

__global__ void d4(float* dest, float** ADDR, int H, int W){
	float z, za, zb, zc, zd, zab, zac, zad, zbc, zbd, zcd, zabc, zabd, zacd, zbcd, zabcd;
	float q, sI, sII, sIII, sIV;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){

		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zd = *(ADDR[4] + tid);
		zab = *(ADDR[5] + tid);
		zac = *(ADDR[6] + tid);
		zad = *(ADDR[7] + tid);
		zbc = *(ADDR[8] + tid);
		zbd = *(ADDR[9] + tid);
		zcd = *(ADDR[10] + tid);
		zabc = *(ADDR[11] + tid);
		zabd = *(ADDR[12] + tid);
		zacd = *(ADDR[13] + tid);
		zbcd = *(ADDR[14] + tid);
		zabcd = *(ADDR[15] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q*(1.0f - q);
		sII = -sI*(1.0f - 2.0f*q);
		sIII = sI*(1.0f - 6.0f*sI);
		sIV = sII*(1.0f - 12.0f*sI);

		dest[tid] = 
			sI*zabcd + 
			sII*(zacd*zb + zad*zbc + za*zbcd + zac*zbd + zabd*zc + zab*zcd + zabc*zd) + 
			sIII*(zad*zb*zc + za*zbd*zc + za*zb*zcd + zac*zb*zd + za*zbc*zd + zab*zc*zd) + 
			sIV*za*zb*zc*zd;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d3(float* dest, float** ADDR, int H, int W){
	float z, za, zb, zc, zab, zac, zbc, zabc;
	float q, sI, sII, sIII;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){
		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zc = *(ADDR[3] + tid);
		zab = *(ADDR[4] + tid);
		zac = *(ADDR[5] + tid);
		zbc = *(ADDR[6] + tid);
		zabc = *(ADDR[7] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q*(1.0f - q);
		sII = -sI*(1.0f - 2.0f*q);
		sIII = sI*(1.0f - 6.0f*sI);

		dest[tid] = sI*zabc + sII*(zac*zb +za*zbc + zab*zc) + sIII*za*zb*zc;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d2(float* dest, float** ADDR, int H, int W){
	float z, za, zb, zab;
	float q, sI, sII;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){
		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);
		zb = *(ADDR[2] + tid);
		zab = *(ADDR[3] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q*(1.0f - q);
		sII = -sI*(1.0f - 2.0f*q);

		dest[tid] = sI*zab + sII*za*zb;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d1(float* dest, float** ADDR, int H, int W){
	float z, za;
	float q, sI;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){
		z = *(ADDR[0] + tid);
		za = *(ADDR[1] + tid);

		q = 1.0f / (1.0f + expf(z));
		sI = q*(1.0f - q);

		dest[tid] = sI*za;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d0(float* dest, float** ADDR, int H, int W){
	float z;

	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < H*W){
		z = *(ADDR[0] + tid);
		dest[tid] = 1.0f / (1.0f + expf(-z));
		tid += blockDim.x * gridDim.x;
	}
}

/*
 * Generic costs live in cost_kernels.cuh. Each cost supplies its scalar loss
 * and output-derivative adjoints to the retained backpropagation kernels.
 */

__global__ void Resile_k(int hW, int wW, float* W, float* delta_W, float* act_grad_W, float* inact_grad_W) {
	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float nW;
	while (tid < hW*wW) {

		if (act_grad_W[tid] * inact_grad_W[tid] > 0) delta_W[tid] *= 1.2f;

		nW = W[tid] - copysignf(delta_W[tid], act_grad_W[tid]);

		if (act_grad_W[tid] * inact_grad_W[tid] < 0) {
			W[tid] += copysignf(delta_W[tid], inact_grad_W[tid]); // rolling back previous step
			delta_W[tid] *= 0.5f;
			act_grad_W[tid] = 0.0f;
		}
		else if (act_grad_W[tid] != 0.0f) {
			if (fabsf(nW) <= 20.0f) W[tid] = nW;
			else {
				// No step was applied: restart at 0.1 and prevent a false rollback.
				delta_W[tid] = 0.1f;
				act_grad_W[tid] = 0.0f;
			}
		}
		
		tid += blockDim.x * gridDim.x;
	}

}

__global__ void ADAM_k(int h, int w, float* W, float* grad, float* M, float* V, float alpha, float beta1, float beta2, float beta1t, float beta2t) {
	
	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float nW;
	float M_hat;
	float V_hat;
	while (tid < h * w) {

		M[tid] = beta1 * M[tid] + (1 - beta1) * grad[tid];
		V[tid] = beta2 * V[tid] + (1 - beta2) * grad[tid] * grad[tid];
		M_hat = M[tid]/(1 - beta1t);
		V_hat = V[tid]/(1 - beta2t);

		nW = W[tid] - alpha * M_hat/(sqrtf(V_hat) + 1e-8);
		if(fabsf(nW) <= 20.0f) W[tid] = nW;
		
		tid += blockDim.x * gridDim.x;
	}

}

__global__ void Reset_Resile_k(int hW, int wW, float R, float* delta_W,
                              float* act_grad_W, float* inact_grad_W) {
    long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < hW*wW) {
        if (delta_W[tid] < R) {
            delta_W[tid] = R;
            // The restored step was not used for the preceding update.
            // Clear only this entry's history to prevent a spurious rollback.
            act_grad_W[tid] = 0.0f;
            inact_grad_W[tid] = 0.0f;
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void add_noise(int h, int w, float* M, float* N) {
	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < h*w) {
		M[tid] = M[tid] + N[tid];
	//	M[tid]=fmaxf(0.0f, fminf(1.0f, M[tid]));
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void Set_Resile_k(int hW, int wW, float R, float* delta_W, float* current, float* previous) {
	long unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < hW*wW) {
		delta_W[tid] = R;
        current[tid] = previous[tid] = 0.0f;
		tid += blockDim.x * gridDim.x;
	}
}

// Device-side constant initialization; random weights are supplied by Python.
__global__ void Fill_Constant_k(int count, float value, float* dest) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        dest[i]=value;
}
