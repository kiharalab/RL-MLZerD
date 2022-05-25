#ifndef _CHARMM_H_
#define _CHARMM_H_

using namespace std;

#include <string>
#include <cstdio>

const string CHARMM_ATOM_TYPES = "charmm_atom_types.in";
const string CHARMM_ATOM_PARAMS = "charmm_params.in";

class charmm_type
{
	public:
		string RES;
		string PDBTYPE;
		string CHARMMTYPE;
		double charge;
		double atomrad;
		int acptype;

		charmm_type(string s1, string s2, string s3, double c1, double c2, int i1)
		{
			RES = s1;
			PDBTYPE = s2;
			CHARMMTYPE = s3;
			charge = c1;
			atomrad = c2;
			acptype = i1;
		}

		charmm_type()
		{
			RES = "";
			PDBTYPE = "";
			CHARMMTYPE = "";
			charge = 0.0;
			atomrad = 1.9;
			acptype = 0;
		}
};

class charmm_param
{
	public:
		double RMIN;
		double AVOL;
		double DGFREE;
		double DGREF;
		double LAMBDA;
		double EMIN;
		bool   valset;

		charmm_param(double d1, double d2, double d3, double d4, double d5, double d6)
		{
			AVOL = d1;
			DGREF = d2;
			DGFREE = d3;
			LAMBDA = d4;
			RMIN = d5;
			EMIN = d6;
			valset = true;
		}
		charmm_param()
		{
			AVOL = 0.;
			DGREF = 0.;
			DGFREE = 0.;
			LAMBDA = 0.;
			RMIN = 0.;
			EMIN = 0.;
			valset = false;
		}

		string tostring()
		{
			char buf[200];
			sprintf(buf, "%f %f %f %f %f %f", AVOL, DGREF, DGFREE, LAMBDA, RMIN, EMIN);
			return string(buf);
		}
};

#endif
