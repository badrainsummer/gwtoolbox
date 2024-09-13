/****

@author  Jon Gair, Stas Babak 2010

***/

#include "FreqAK_RA.hh"


void EccentricLISAMotion(float kappa0, float lambda0, double t, double* R, \
	                         double** q, double** n){
	//	double Omega = LISAWP_TWOPI/year;
                double Omega = LISAWP_TWOPI*3.1709791983764586e-8;
		double alpha = Omega*t + kappa0;
		double xi[3];
		xi[0] = lambda0;             //////////  MOVE THOSE CALCULATIONS OUT TO THE CONSTRUCTOR
		xi[1] = lambda0 + 2.0*LISAWP_PI/3.0;
		xi[2] = lambda0 + 4.0*LISAWP_PI/3.0;
		double RAU = LISAWP_AU_SI/LISAWP_C_SI;

		R[0] = RAU*cos(alpha);
		R[1] = RAU*sin(alpha);
		R[2] = 0;

		for(int i=0; i<3; i++){ // loop over s/c
		     q[i][0] = ( sin(alpha)*cos(alpha)*sin(xi[i]) - \
    				(1.+ sin(alpha)*sin(alpha))*cos(xi[i])  )/(2.0*sqrt(3.));
                     q[i][1] = ( sin(alpha)*cos(alpha)*cos(xi[i]) - \
    				(1.+ cos(alpha)*cos(alpha))*sin(xi[i])  )/(2.0*sqrt(3.));
                     q[i][2] = -0.5*cos(alpha - xi[i]);
		}
		for(int i=0; i<3; i++){ // links: 1st index - s/c, 2nd index - coordinates
                    n[0][i] = q[1][i] - q[2][i];
		    n[1][i] = q[2][i] - q[0][i];
		    n[2][i] = q[0][i] - q[1][i];
		}
}


FreqAK_RA::FreqAK_RA(double timestep, double maxDur, double dtPhase){

      dt_w = timestep;
      Tobs = maxDur;
      Nps = (int)floor(Tobs/dt_w);
      df = 1./Tobs;
      dt_ph = dtPhase;
      Nph = (int)floor(Tobs/dt_ph);
      arm =  LISA_arm/ LISAWP_C_SI; // LISA's arm in sec

      phiev = NULL;
   	alpev = NULL;
   	gamev = NULL;
   	eccev = NULL;
   	nuev = NULL;
   	gamdotev =  NULL;
   	alpdotev = NULL;
   	phiddotev = NULL;
   	gamddotev = NULL;
   	alpddotev = NULL;
   	Ampev = NULL;
   	FplusI = NULL;
      FplusII = NULL;
      FcrosI = NULL;
      FcrosII = NULL;
      tm_ph = NULL;

      Xp = NULL;
      Xc = NULL;
      Yp = NULL;
      Yc = NULL;
      Zp = NULL;
      Zc = NULL;

      phiev = new double[Nph];
      tm_ph = new double[Nph];
      alpev = new double[Nph];
      gamev = new double[Nph];
      eccev = new double[Nph];
      nuev = new double[Nph];
      gamdotev = new double[Nph];
      alpdotev = new double[Nph];
      phiddotev = new double[Nph];
      gamddotev = new double[Nph];
      alpddotev = new double[Nph];
      Ampev = new double[Nph];


      FplusI = new double[Nph];
      FcrosI = new double[Nph];
      FplusII = new double[Nph];
      FcrosII = new double[Nph];

      Xp = new std::complex<double>*[25];
      Xc = new std::complex<double>*[25];
      Yp = new std::complex<double>*[25];
      Yc = new std::complex<double>*[25];
      Zp = new std::complex<double>*[25];
      Zc = new std::complex<double>*[25];
      for (int i=0; i<25; i++){
            Xp[i] = new std::complex<double>[Nph];
            Xc[i] = new std::complex<double>[Nph];
            Yp[i] = new std::complex<double>[Nph];
            Yc[i] = new std::complex<double>[Nph];
            Zp[i] = new std::complex<double>[Nph];
            Zc[i] = new std::complex<double>[Nph];
      }

}

FreqAK_RA::~FreqAK_RA(){

   if (phiev != NULL)
      delete [] phiev;
	if (alpev != NULL)
	    delete [] alpev;
	if	(gamev != NULL)
	    delete gamev;
	if (eccev != NULL)
	   delete [] eccev;
	if (nuev != NULL)
	   delete [] nuev;
	if (gamdotev !=  NULL)
	   delete [] gamdotev;
	if (alpdotev != NULL)
	   delete [] alpdotev;
	if (phiddotev != NULL)
	   delete [] phiddotev;
	if (gamddotev != NULL)
	   delete [] gamddotev;
	if (alpddotev != NULL)
	   delete [] alpddotev;
	if (Ampev != NULL)
	   delete [] Ampev;
	if (FplusI != NULL)
	   delete [] FplusI;
   if (FplusII != NULL)
      delete [] FplusII;
   if (FcrosI != NULL)
      delete [] FcrosI;
   if (FcrosII != NULL)
      delete [] FcrosII;

   if (tm_ph != NULL)
     delete [] tm_ph;

  for (int i=0; i<25; i++){
     delete [] Xp[i];
     delete [] Xc[i];
     delete [] Yp[i];
     delete [] Yc[i];
     delete [] Zp[i];
     delete [] Zc[i];
  }
  delete [] Xp;
  delete [] Xc;
  delete [] Yp;
  delete [] Yc;
  delete [] Zp;
  delete [] Zc;
}


void FreqAK_RA::PhaseEv_RA(EMRItemplate& S, double t0, double dur, double tStart){

   double k[3];
   double uhat[3];
   double vhat[3];
   double u[3];
   double v[3];
   double kn[3];
   double kq[3];

   double up = (S.ctS*S.stK*cos(S.phS - S.phK) - S.ctK*S.stS);
   double dw = (S.stK*sin(S.phS-S.phK));
   double psi;
   if (dw != 0.0) {
      psi = atan2(up, dw);
   }else {
      psi = 0.5*LISAWP_PI;
   }
   double c2psi=cos(2.*psi);
   double s2psi=sin(2.*psi);


   // note that k = -n, where k is propagation vector and n is sky location
   k[0] =  -S.stS*S.cpS;
   k[1] = -S.stS*S.spS;
   k[2] = -S.ctS;

   uhat[0] = S.ctS*S.cpS;
   uhat[1] = S.ctS*S.spS;
   uhat[2] = -S.stS;

   vhat[0] = S.spS;
   vhat[1] = -S.cpS;
   vhat[2] = 0.0;
   double nU, nV;

   S.t0 = 0.0;
   double e0 = S.e0;
   double nu0 = S.nu0;
   double ph0 = S.Phi0;
   double al0 =  S.alpha0;
   double gam0 = S.gamma0;

   double* R;
   double** q;
   double** n;
   R = new double[3];
   q = new double*[3];
   n = new double*[3];
   for (int i=0; i<3; i++){
       q[i] = new double[3];
       n[i] = new double[3];
   }

   double AUsec =  LISAWP_AU_SI/LISAWP_C_SI;
   double clam=cos(S.lam);
   double slam=sin(S.lam);



   double M = S.Mt;
   double mu = S.mt;
   double e = S.e0;
   double e2=e*e;
   double Sp = S.a;
   double nu = S.nu0;
   double phi = S.Phi0;
   double gam = S.gamma0;
   double alp = S.alpha0;
   double Y, Z;
   double edotm, nudotm, phidotm, alpdotm, gamdotm;
   double edot, nudot, phidot, alpdot, gamdot;
   double dalpdnu, dalpde, dgamdnu, dgamde, alpddot, gamddot, phiddot;
   double de, dnu, dphi, dgam, dalp, rhs;
   double T=0.0;
   int imax, ind;
   double ampfct= S.Ampl;

   double nn, mm, om;
   std::complex<double> chi0;
   std::complex<double> chi1;
   std::complex<double> chi2;
   std::complex<double> img(0.0, 1.0);
   double x, x2;

   std::cout << "starting the loop\n";
   for (int i=0; i<Nph; i++){

          e2=e*e;
          Y=1./(1.-e2);
          Z=pow(LISAWP_TWOPI*M*nu,1./3.);
          if (i != 0){
             edotm=edot;
             nudotm=nudot;
             phidotm=phidot;
             alpdotm=alpdot;
             gamdotm=gamdot;
          }
          edot = e*mu/M/M*(-1./15.*pow(Y,3.5)*pow(Z,8.)*((304.+121.*e2)/Y+Z*Z*(70648.-231960.*e2-56101.*e2*e2)/56.)+\
                   Sp*clam*pow(Z,11.)*pow(Y,4.)*(8184.+10064.*e2+789.*e2*e2)/30.);
          nudot = 96./(10.*M_PI)*mu/pow(M,3)*(pow(Z,11.)*pow(Y,4.5)*((96.+292.*e2+37.*e2*e2)/Y/96.\
       			       +Z*Z*(20368.-61464.*e2-163170.*e2*e2-13147.*e2*e2*e2)/5376.) - \
       			      pow(Z,14.)*pow(Y,5.)*Sp*clam*(1168.+9688.*e2+6286.*e2*e2 +195.*e2*e2*e2)/192.);
          phidot = 2.*LISAWP_PI*nu;
          alpdot = 8.*LISAWP_PI*LISAWP_PI*nu*nu*Sp*M*pow(Y,1.5);
          gamdot = 6.*LISAWP_PI*nu*Z*Z*Y*(1.+.25*Z*Z*Y*(26.-15.*e2))-3.*clam*alpdot;

          dalpdnu=16.*Sp*LISAWP_PI*nu*M*pow(Y,1.5);
          dalpde=12.*Sp*LISAWP_PI*nu*nu*M*sqrt(Y)*2.*e*Y*Y;
          dgamdnu=6.*Z*Z*Y*(1.+.25*Z*Z*Y*(26.-15.*e2))-3.*clam*dalpdnu+12.*Z*Y*(1.+.5*Z*Z*Y*(25.-15.*e2))*Z/(3.);
          dgamde=(6.*nu*Z*Z*(1.+.5*Z*Z*Y*(26.-15.*e2)))*2.*e*Y*Y-45.*nu*Z*Z*Z*Z*Y*Y*e-3.*clam*dalpde;

          alpddot=LISAWP_PI*(dalpdnu*nudot+dalpde*edot);
          gamddot=LISAWP_PI*(dgamdnu*nudot+dgamde*edot);
          phiddot=LISAWP_TWOPI*nudot;

          if (i == 0) {
             edotm=edot;
             nudotm=nudot;
             alpdotm=alpdot;
             gamdotm=gamdot;
             phidotm=phidot;
          }

          de=(1.5*edot-.5*edotm)*dt_ph;
          dnu=(1.5*nudot-.5*nudotm)*dt_ph;
          dphi=(1.5*phidot-.5*phidotm)*dt_ph;
          dgam=(1.5*gamdot-.5*gamdotm)*dt_ph;
          dalp=(1.5*alpdot-.5*alpdotm)*dt_ph;

          phiev[i]=phi;
          alpev[i]=alp;
          gamev[i]=gam;
          eccev[i]=e;
          nuev[i]=nu;
          gamdotev[i]=gamdot;
          alpdotev[i]=alpdot;
          Ampev[i]=ampfct*pow(LISAWP_TWOPI*S.Mt*nuev[i], 2./3.);
          phiddotev[i]=phiddot;
          gamddotev[i]=gamddot;
          alpddotev[i]=alpddot;
          tm_ph[i] = T;
          e+=de;
          phi+=dphi;
          gam+=dgam;
          alp+=dalp;
          e2=e*e;
          nu+=dnu;
          T+= dt_ph;

          rhs = pow( (1.0-e2)/(6.0+2.0*e), 1.5 )/(LISAWP_TWOPI * M);
          if(rhs - nu <= 0.0){
             std::cout << "*** we reached plunge at t = " << T << std::endl;
             std::cout << " i = " << i << std::endl;
             imax = i;
             S.tPl = T;
             S.e_pl = e;
             S.nu_pl = nu;
             S.alpha_pl = alp;
             S.gamma_pl = gam;
             S.Phi_pl = phi;
        	    break;
          }
          // LISA's motion
          //  EccentricLISAMotion(float kappa0, float lambda0, double t, double* R, \
	                         double** q, double** n)
          EccentricLISAMotion(0.0, 0.0, T, R, q, n);
          for(int j =0; j<3; j++){
       	      kn[j] = 0.0;
               kq[j] = 0.0;
       		   nU = 0.0;
       		   nV = 0.0;
       		   for(int ii=0; ii<3; ii++){
       		   	 kn[j] += k[ii]*n[j][ii];
       		 	 kq[j] += k[ii]*q[j][ii];
       			 nU += uhat[ii]*n[j][ii];
       			 nV += vhat[ii]*n[j][ii];
       		   }
       		   u[j] = 0.5*(nU*nU - nV*nV);
       		   v[j] = nU*nV;
       	 }

          ind = 0;
          for (int ii=0; ii<5; ii++){ // nn harmonic
              nn = (double)ii+1.;
              for (int jj=0; jj<5; jj++){
                 mm = (double)jj-2.;
                 om = nn*LISAWP_TWOPI*nu + 2.*gamdot + mm*alpdot;
                 x = om*arm;
                 x2 = 0.5*x;
                 chi1 = -x*sin(x)*( Sinc(x2*(1.-kn[1]))*exp(-img*x) \
                                    + Sinc(x2*(1.+kn[1])) )*exp(-img*x2*(3.0 + kq[0] + kq[2]));
                 chi2 = x*sin(x)*( Sinc(x2*(1.-kn[2])) + exp(-img*x)*\
                 							Sinc(x2*(1.+kn[2])) )*exp(-img*x2*(3.0 + kq[1] + kq[0]));

                 Xp[ind][i] = (u[1]*c2psi - v[1]*s2psi)*chi1 + (u[2]*c2psi - v[2]*s2psi)*chi2;
                 Xc[ind][i] = (v[1]*c2psi + u[1]*s2psi)*chi1 + (v[2]*c2psi + u[2]*s2psi)*chi2;


                 chi2 = -x*sin(x)*( Sinc(x2*(1.-kn[2]))*exp(-img*x) \
                                      + Sinc(x2*(1.+kn[2])) )*exp(-img*x2*(3.0 + kq[1] + kq[0]));
                 chi0 = x*sin(x)*( Sinc(x2*(1.-kn[0])) + exp(-img*x)*\
                   							Sinc(x2*(1.+kn[0])) )*exp(-img*x2*(3.0 + kq[2] + kq[1]));

                 Yp[ind][i] = (u[2]*c2psi - v[2]*s2psi)*chi2 + (u[0]*c2psi - v[0]*s2psi)*chi0;
                 Yc[ind][i] = (v[2]*c2psi + u[2]*s2psi)*chi2 + (v[0]*c2psi + u[0]*s2psi)*chi0;


                 chi0 = -x*sin(x)*( Sinc(x2*(1.-kn[0]))*exp(-img*x) \
                                        + Sinc(x2*(1.+kn[0])) )*exp(-img*x2*(3.0 + kq[2] + kq[1]));
                 chi1 = x*sin(x)*( Sinc(x2*(1.-kn[1])) + exp(-img*x)*\
                     							Sinc(x2*(1.+kn[1])) )*exp(-img*x2*(3.0 + kq[0] + kq[2]));

                 Zp[ind][i] = (u[0]*c2psi - v[0]*s2psi)*chi0 + (u[1]*c2psi - v[1]*s2psi)*chi1;
                 Zc[ind][i] = (v[0]*c2psi + u[0]*s2psi)*chi0 + (v[1]*c2psi + u[1]*s2psi)*chi1;


       //          fout03  << T << spr << om << spr  << tp.real() << spr << tp.imag() << std::endl; //<< "   " << Xp[ind][i].imag() << "   " << \
                             Xc[ind][i].real() << "   " << Xc[ind][i].imag() << "   ";
                 ind++;

              }
          }
     }
   delete [] R;
   for (int i=0; i<3; i++){
       delete [] q[i];
       delete [] n[i];
   }
   delete [] q;
   delete [] n;

}

// TODO interface is different now, we do not us MAtrix anymore!!!!
void FreqAK_RA::ConstructWaveFreq_RA(EMRItemplate& S, double* Xf_r, double* Xf_im,  double* Yf_r, double* Yf_im,
                        double* Zf_r, double* Zf_im)

{

     // NOTE!!!! Xf, Yf, Zf must have proper size and be zero

     double T = 0.0;
     double ec, ec2, ec3, ec4, ec5, ec6, ec7, hamp;
     int ind_low, ind_up;
     double delta, eps, amp;
     double  xi, Apc, Aps, Acc, Acs;
     int harmms[]={-2,-1,0,1,2};
     double fact;
     std::complex<double> hplus, hcross;
     std::complex<double> cFp;
     std::complex<double> cFc;
     std::complex<double> x, xp, xc;
     std::complex<double> img(0.0, 1.0);
     double om_in;
     double om_fin;
     double dOm = df*LISAWP_TWOPI;
     double Om, dom, delom;
     double xi_in, xi_fin, dxi;
     double orbOm = LISAWP_TWOPI/LISAWP_YRSID_SI;
     double AUsec =  LISAWP_AU_SI/LISAWP_C_SI;
     double DM = AUsec*S.stS;
     double faza, sinph, cosph;
     int ind;
     double nn,mm;
     std::string spr = "    ";

     double cS= S.ctS;
     double sS= S.stS;
     double cK= S.ctK;
     double sK= S.stK;
     double cSp= S.cpS;
     double sSp= S.spS;
     double cKp= S.cpK;
     double sKp= S.spK;
     double clam=cos(S.lam);
     double slam=sin(S.lam);
     double Sn = cS*cK + sS*sK*cos(S.phS  - S. phK);
     double cX = Sn;

     double sX = sqrt( sS*sS*cK*cK - 2.*sS*sSp*cK*cS*sK*sKp + \
                      cS*cS*sK*sK - 2.*cS*sK*cKp*sS*cSp*cK + \
                      sS*sS*cSp*cSp*sK*sK*sKp*sKp - 2.*sS*sS*cSp*sK*sK*sKp*sSp*cKp +\
                      sS*sS*sSp*sSp*sK*sK*cKp*cKp);

     double Apc1, Aps1, Apcn1, Apc2, Aps2, Apcn2, Aqc1, Aqs1, Aqcn1, Aqc2, Aqs2, Aqcn2;

       Apc1 = ( -cK*cKp*sS*cSp - cK*sKp*sS*sSp + sK*cS )/(sX);
       Aps1 = ( sKp*sS*cSp - cKp*sS*sSp )/(sX);
       Apcn1 = ( cK*cS + sK*cKp*sS*cSp + sK*sKp*sS*sSp - cX)*clam/(sX*slam);

       Apc2 = (sS*cSp*sKp - sS*sSp*cKp )*clam/(sX);
       Aps2 = ( cK*cKp*sS*cSp + cK*sKp*sS*sSp - cS*sK )*clam/(sX);
       Apcn2 = 0.0;


       Aqc1 = ( sS*cSp*sKp - sS*sSp*cKp  )*cX/(sX);
       Aqs1 = ( cK*cKp*sS*cSp + cK*sKp*sS*sSp - cS*sK )*cX/(sX);
       Aqcn1 = 0.0;


       Aqc2 = cX*clam*( cK*cKp*sS*cSp + cK*sKp*sS*sSp - sK*cS)/(sX);
       Aqs2 = -cX*clam*sS*( sKp*cSp - cKp*sSp )/(sX);
       Aqcn2 = -( cX*clam*clam*( cK*cS + sK*cKp*sS*cSp + sK*sKp*sS*sSp ) + \
                                1.- cX*cX - clam*clam )/(sX*slam);


     double Bp1c1 = 2.0*(Apc1*Apcn1 - Aqc1*Aqcn1 + Aqc2*Aqcn2 - Apc2*Apcn2);
     double Bp1c2 =  0.5*(Aps2*Aps2 - Aqc1*Aqc1  + Apc1*Apc1  - Aps1*Aps1 + \
                                Aqc2*Aqc2 + Aqs1*Aqs1 - Apc2*Apc2 - Aqs2*Aqs2);
     double Bp1s1 = 2.0*(Aqs2*Aqcn2 - Aps2*Apcn2 - Aqs1*Aqcn1 + Aps1*Apcn1);
     double Bp1s2 = (Apc1*Aps1 + Aqc2*Aqs2 - Apc2*Aps2 - Aqc1*Aqs1);
     double Bp1cn = 0.5*(Apc1*Apc1 + Aps1*Aps1 - Aqc1*Aqc1 - Aqs1*Aqs1 - Apc2*Apc2 \
                                + Aqc2*Aqc2 + Aqs2*Aqs2 - Aps2*Aps2) + Aqcn2*Aqcn2 - Aqcn1*Aqcn1 \
                                + Apcn1*Apcn1 - Apcn2*Apcn2;

     double Bp2c1 = (Apcn1*Apc2 + Apc1*Apcn2 - Aqcn1*Aqc2 - Aqc1*Aqcn2);
     double Bp2c2 = 0.5*(Aqs1*Aqs2 - Aps1*Aps2 + Apc1*Apc2 - Aqc1*Aqc2);
     double Bp2s1 = (Aps1*Apcn2 + Apcn1*Aps2 - Aqcn1*Aqs2 - Aqs1*Aqcn2);
     double Bp2s2 = 0.5*( Apc1*Aps2 - Aqc1*Aqs2 + Aps1*Apc2 - Aqs1*Aqc2);
     double Bp2cn = 0.5*(Aps1*Aps2 - Aqs1*Aqs2 - Aqc1*Aqc2 + Apc1*Apc2) -Aqcn1*Aqcn2 + Apcn1*Apcn2;

     double Bc1c1 = (-Apc2*Aqcn2 - Apcn2*Aqc2 + Apc1*Aqcn1 + Apcn1*Aqc1);
     double Bc1c2 = 0.5*( Apc1*Aqc1 - Aps1*Aqs1 - Apc2*Aqc2 + Aps2*Aqs2);
     double Bc1s1 = (Apcn1*Aqs1 - Aps2*Aqcn2 + Aps1*Aqcn1 - Apcn2*Aqs2);
     double Bc1s2 = 0.5*(-Apc2*Aqs2 + Apc1*Aqs1 - Aps2*Aqc2 + Aps1*Aqc1);
     double Bc1cn = -Apcn2*Aqcn2 + Apcn1*Aqcn1 + 0.5*(Apc1*Aqc1 - Aps2*Aqs2 + Aps1*Aqs1 - Apc2*Aqc2);

     double Bc2c1 = (Aqc1*Apcn2 + Aqcn1*Apc2 + Apc1*Aqcn2 + Apcn1*Aqc2);
     double Bc2c2 = 0.5*( Apc1*Aqc2 - Aps1*Aqs2 + Aqc1*Apc2 - Aqs1*Aps2);
     double Bc2s1 = (Apcn1*Aqs2 + Aqs1*Apcn2 + Aps1*Aqcn2 + Aqcn1*Aps2);
     double Bc2s2 = 0.5*(Aqc1*Aps2 + Apc1*Aqs2 + Aqs1*Apc2 + Aps1*Aqc2);
     double Bc2cn = Aqcn1*Apcn2 + Apcn1*Aqcn2 + 0.5*(Apc1*Aqc2 + Aqs1*Aps2 +Aps1*Aqs2 + Aqc1*Apc2);

     double AApcos[5],AApsin[5],AAccos[5],AAcsin[5];
       AApcos[0]=0.5*(Bp1c2+Bp2s2);
       AApsin[0]=0.5*(Bp2c2-Bp1s2);
       AAccos[0]=0.5*(Bc1c2+Bc2s2);
       AAcsin[0]=0.5*(Bc2c2-Bc1s2);
       AApcos[1]=0.5*(Bp1c1+Bp2s1);
       AApsin[1]=0.5*(Bp2c1-Bp1s1);
       AAccos[1]=0.5*(Bc1c1+Bc2s1);
       AAcsin[1]=0.5*(Bc2c1-Bc1s1);
       AApcos[2]=Bp1cn;
       AApsin[2]=Bp2cn;
       AAccos[2]=Bc1cn;
       AAcsin[2]=Bc2cn;
       AApcos[3]=0.5*(Bp1c1-Bp2s1);
       AApsin[3]=0.5*(Bp2c1+Bp1s1);
       AAccos[3]=0.5*(Bc1c1-Bc2s1);
       AAcsin[3]=0.5*(Bc2c1+Bc1s1);
       AApcos[4]=0.5*(Bp1c2-Bp2s2);
       AApsin[4]=0.5*(Bp2c2+Bp1s2);
       AAccos[4]=0.5*(Bc1c2-Bc2s2);
       AAcsin[4]=0.5*(Bc2c2+Bc1s2);
       int ki;

       std::complex<double> test;
      //std::ofstream fout09("Data/Scratch.dat");
     for (int i=1; i<Nph; i++){
        xi_in = tm_ph[i-1] - DM*cos(orbOm*tm_ph[i-1]-S.phS);
        xi_fin = tm_ph[i] - DM*cos(orbOm*tm_ph[i]-S.phS);
        dxi = xi_fin - xi_in;
        ind = 0;
        // loops over harmonics
        for (int j=0;j<5;j++) {
        	  nn=(double)(j+1);
        	  for(int jj=0; jj<5; jj++){
        	     mm = (double)jj-2.;
        	     om_in = nn*LISAWP_TWOPI*nuev[i-1] + 2.*gamdotev[i-1]  + mm*alpdotev[i-1];
        	     om_fin = nn*LISAWP_TWOPI*nuev[i] + 2.*gamdotev[i]  + mm*alpdotev[i];
              delom = om_fin - om_in;
              ind_low = (int) ceil(om_in/dOm);
              ind_up = (int) floor(om_fin/dOm);
              if (om_fin == (double)ind_up*dOm && om_fin != 0.){
                  std::cout << "Fourier freq = harm freq\n" << om_fin << spr << (double)ind_up*dOm << std::endl;
                  ind_up = ind_up-1; // otherwise we count twice this bin
              }
              if (om_in != 0. && om_fin != 0.){
                 // loop over fourier bins between two values of harmonic
                 for (int ii=ind_low; ii<=ind_up; ii++){
                     Om = (double)ii * dOm;
                     delta = (Om - om_in)/delom;
                     eps = 1.-delta;
                     T =  tm_ph[i]*delta + tm_ph[i-1]*eps;
                     xi = T - DM*cos(orbOm*T - S.phS);
                     delta = (xi - xi_in)/dxi;
                     eps = 1.-delta;
                     dom = (nn*phiddotev[i-1] + 2.*gamddotev[i-1]  + mm*alpddotev[i-1])*eps +
                           (nn*phiddotev[i] + 2.*gamddotev[i]  + mm*alpddotev[i])*delta;
                     faza = (nn*phiev[i-1] + 2.*gamev[i-1]  + mm*alpev[i-1])*eps +
                               (nn*phiev[i] + 2.*gamev[i]  + mm*alpev[i])*delta;
                     amp = Ampev[i-1]*eps + Ampev[i]*delta;
                     ec = eccev[i-1]*eps +  eccev[i]*delta;
                     ec2=ec*ec;
                     ec3=ec2*ec;
                 	   ec4=ec3*ec;
                     ec5=ec4*ec;
                     ec6=ec5*ec;
                     ec7=ec6*ec;
                     switch (j+1) {
               			   case 1:
               				   hamp =3.*ec-1.625*ec3;
               				   break;
               			   case 2:
               				   hamp =-4.+10.*ec2-5.75*ec4;
               				   break;
               			   case 3:
               				   hamp =-9.*ec+21.375*ec3-15.046875*ec5;
               				   break;
               			   case 4:
               				   hamp =-16.*ec2+40.*ec4-101.*ec6/3.;
               				   break;
               			   case 5:
               				   hamp =-625.*ec3/24.+26875.*ec5/384.-210625.*ec7/3072.;
               				   break;
               		}
               		delta = (T - tm_ph[i-1])/dt_ph;
                     eps = 1.-delta;
                     cFp = Xp[ind][i-1]*eps + Xp[ind][i]*delta;
                     cFc = Xc[ind][i-1]*eps + Xc[ind][i]*delta;

                     Apc = hamp*AApcos[harmms[jj]+2];
         		      Aps = hamp*AApsin[harmms[jj]+2]; // should be "-" for strain
         		      Acc = 2.*hamp*AAccos[harmms[jj]+2];
         		      Acs = 2.*hamp*AAcsin[harmms[jj]+2];

         		      sinph = sin(faza - Om*xi + LISAWP_PI*0.25);
                     cosph = cos(faza - Om*xi + LISAWP_PI*0.25);

                     fact = 0.5*amp*sqrt(LISAWP_TWOPI/dom);
                     //hplus = Apc - img*Aps;
                     //hcross = Acc - img*Acs;
                    // Xf[ii] = Xf[ii] + fact*(cFp*hplus + cFc*hcross)*(cosph + img*sinph);
                     ki = Nps - ii;
                     //xp = fact*hplus*(cosph + img*sinph);
                     //xc = fact*hcross*(cosph + img*sinph);
                     xp = fact*(Apc*cosph + Aps*sinph + img*(Apc*sinph - Aps*cosph));
                     xc = fact*(Acc*cosph + Acs*sinph + img*(Acc*sinph - Acs*cosph));
                     // TODO output was changed
                     // -------------------------------------------------------------
                     //Xf(ii) += (cFp*xp + cFc*xc).real();
                     //Xf(ki) += (cFp*xp + cFc*xc).imag();
                     Xf_r[ii] += (cFp*xp + cFc*xc).real();
                     Xf_im[ii] += (cFp*xp + cFc*xc).imag();
                     cFp = Yp[ind][i-1]*eps + Yp[ind][i]*delta;
                     cFc = Yc[ind][i-1]*eps + Yc[ind][i]*delta;
                     Yf_r[ii] += (cFp*xp + cFc*xc).real();
                     Yf_im[ii] += (cFp*xp + cFc*xc).imag();
                     //Yf(ii) += (cFp*xp + cFc*xc).real();
                     //Yf(ki) += (cFp*xp + cFc*xc).imag();
                     cFp = Zp[ind][i-1]*eps + Zp[ind][i]*delta;
                     cFc = Zc[ind][i-1]*eps + Zc[ind][i]*delta;
                     Zf_r[ii] += (cFp*xp + cFc*xc).real();
                     Zf_im[ii] += (cFp*xp + cFc*xc).imag();
                     //Zf(ii) += (cFp*xp + cFc*xc).real();
                     //Zf(ki) += (cFp*xp + cFc*xc).imag();
                  }
              }
              ind ++;
        	  }// jj loop
     	  } //j loop
     }
     //fout09.close();
}


void FreqAK_RA::GetHarmonicsXYZ(EMRItemplate& S,  double** X_r, double** X_i, double** Y_r, double** Y_i, double** Z_r,
            double** Z_i, double* phi, double* gam, double* alph, double* tim, double** frqs, double** phase){


     // Currently I do not interpolate between the points for which the orbital evolution was computed

     double T = 0.0;
     double ec, ec2, ec3, ec4, ec5, ec6, ec7, hamp;
     int ind_low, ind_up;
     double delta, eps, amp;
     double  xi, Apc, Aps, Acc, Acs;
     int harmms[]={-2,-1,0,1,2};
     double fact;
     std::complex<double> hplus, hcross;
     std::complex<double> cFp;
     std::complex<double> cFc;
     std::complex<double> x, xp, xc;
     std::complex<double> img(0.0, 1.0);
     double om;
     double dOm = df*LISAWP_TWOPI;
     double Om, dom, delom;
     double xi_in, xi_fin, dxi;
     double orbOm = LISAWP_TWOPI/LISAWP_YRSID_SI;
     double AUsec =  LISAWP_AU_SI/LISAWP_C_SI;
     double DM = AUsec*S.stS;
     double faza, sinph, cosph;
     int ind;
     double nn,mm;
     std::string spr = "    ";

     double cS= S.ctS;
     double sS= S.stS;
     double cK= S.ctK;
     double sK= S.stK;
     double cSp= S.cpS;
     double sSp= S.spS;
     double cKp= S.cpK;
     double sKp= S.spK;
     double clam=cos(S.lam);
     double slam=sin(S.lam);
     double Sn = cS*cK + sS*sK*cos(S.phS  - S. phK);
     double cX = Sn;
     std::complex<double> Ampl;


     double sX = sqrt( sS*sS*cK*cK - 2.*sS*sSp*cK*cS*sK*sKp + \
                      cS*cS*sK*sK - 2.*cS*sK*cKp*sS*cSp*cK + \
                      sS*sS*cSp*cSp*sK*sK*sKp*sKp - 2.*sS*sS*cSp*sK*sK*sKp*sSp*cKp +\
                      sS*sS*sSp*sSp*sK*sK*cKp*cKp);

     double Apc1, Aps1, Apcn1, Apc2, Aps2, Apcn2, Aqc1, Aqs1, Aqcn1, Aqc2, Aqs2, Aqcn2;

       Apc1 = ( -cK*cKp*sS*cSp - cK*sKp*sS*sSp + sK*cS )/(sX);
       Aps1 = ( sKp*sS*cSp - cKp*sS*sSp )/(sX);
       Apcn1 = ( cK*cS + sK*cKp*sS*cSp + sK*sKp*sS*sSp - cX)*clam/(sX*slam);

       Apc2 = (sS*cSp*sKp - sS*sSp*cKp )*clam/(sX);
       Aps2 = ( cK*cKp*sS*cSp + cK*sKp*sS*sSp - cS*sK )*clam/(sX);
       Apcn2 = 0.0;


       Aqc1 = ( sS*cSp*sKp - sS*sSp*cKp  )*cX/(sX);
       Aqs1 = ( cK*cKp*sS*cSp + cK*sKp*sS*sSp - cS*sK )*cX/(sX);
       Aqcn1 = 0.0;


       Aqc2 = cX*clam*( cK*cKp*sS*cSp + cK*sKp*sS*sSp - sK*cS)/(sX);
       Aqs2 = -cX*clam*sS*( sKp*cSp - cKp*sSp )/(sX);
       Aqcn2 = -( cX*clam*clam*( cK*cS + sK*cKp*sS*cSp + sK*sKp*sS*sSp ) + \
                                1.- cX*cX - clam*clam )/(sX*slam);


     double Bp1c1 = 2.0*(Apc1*Apcn1 - Aqc1*Aqcn1 + Aqc2*Aqcn2 - Apc2*Apcn2);
     double Bp1c2 =  0.5*(Aps2*Aps2 - Aqc1*Aqc1  + Apc1*Apc1  - Aps1*Aps1 + \
                                Aqc2*Aqc2 + Aqs1*Aqs1 - Apc2*Apc2 - Aqs2*Aqs2);
     double Bp1s1 = 2.0*(Aqs2*Aqcn2 - Aps2*Apcn2 - Aqs1*Aqcn1 + Aps1*Apcn1);
     double Bp1s2 = (Apc1*Aps1 + Aqc2*Aqs2 - Apc2*Aps2 - Aqc1*Aqs1);
     double Bp1cn = 0.5*(Apc1*Apc1 + Aps1*Aps1 - Aqc1*Aqc1 - Aqs1*Aqs1 - Apc2*Apc2 \
                                + Aqc2*Aqc2 + Aqs2*Aqs2 - Aps2*Aps2) + Aqcn2*Aqcn2 - Aqcn1*Aqcn1 \
                                + Apcn1*Apcn1 - Apcn2*Apcn2;

     double Bp2c1 = (Apcn1*Apc2 + Apc1*Apcn2 - Aqcn1*Aqc2 - Aqc1*Aqcn2);
     double Bp2c2 = 0.5*(Aqs1*Aqs2 - Aps1*Aps2 + Apc1*Apc2 - Aqc1*Aqc2);
     double Bp2s1 = (Aps1*Apcn2 + Apcn1*Aps2 - Aqcn1*Aqs2 - Aqs1*Aqcn2);
     double Bp2s2 = 0.5*( Apc1*Aps2 - Aqc1*Aqs2 + Aps1*Apc2 - Aqs1*Aqc2);
     double Bp2cn = 0.5*(Aps1*Aps2 - Aqs1*Aqs2 - Aqc1*Aqc2 + Apc1*Apc2) -Aqcn1*Aqcn2 + Apcn1*Apcn2;

     double Bc1c1 = (-Apc2*Aqcn2 - Apcn2*Aqc2 + Apc1*Aqcn1 + Apcn1*Aqc1);
     double Bc1c2 = 0.5*( Apc1*Aqc1 - Aps1*Aqs1 - Apc2*Aqc2 + Aps2*Aqs2);
     double Bc1s1 = (Apcn1*Aqs1 - Aps2*Aqcn2 + Aps1*Aqcn1 - Apcn2*Aqs2);
     double Bc1s2 = 0.5*(-Apc2*Aqs2 + Apc1*Aqs1 - Aps2*Aqc2 + Aps1*Aqc1);
     double Bc1cn = -Apcn2*Aqcn2 + Apcn1*Aqcn1 + 0.5*(Apc1*Aqc1 - Aps2*Aqs2 + Aps1*Aqs1 - Apc2*Aqc2);

     double Bc2c1 = (Aqc1*Apcn2 + Aqcn1*Apc2 + Apc1*Aqcn2 + Apcn1*Aqc2);
     double Bc2c2 = 0.5*( Apc1*Aqc2 - Aps1*Aqs2 + Aqc1*Apc2 - Aqs1*Aps2);
     double Bc2s1 = (Apcn1*Aqs2 + Aqs1*Apcn2 + Aps1*Aqcn2 + Aqcn1*Aps2);
     double Bc2s2 = 0.5*(Aqc1*Aps2 + Apc1*Aqs2 + Aqs1*Apc2 + Aps1*Aqc2);
     double Bc2cn = Aqcn1*Apcn2 + Apcn1*Aqcn2 + 0.5*(Apc1*Aqc2 + Aqs1*Aps2 +Aps1*Aqs2 + Aqc1*Apc2);

     double AApcos[5],AApsin[5],AAccos[5],AAcsin[5];
       AApcos[0]=0.5*(Bp1c2+Bp2s2);
       AApsin[0]=0.5*(Bp2c2-Bp1s2);
       AAccos[0]=0.5*(Bc1c2+Bc2s2);
       AAcsin[0]=0.5*(Bc2c2-Bc1s2);
       AApcos[1]=0.5*(Bp1c1+Bp2s1);
       AApsin[1]=0.5*(Bp2c1-Bp1s1);
       AAccos[1]=0.5*(Bc1c1+Bc2s1);
       AAcsin[1]=0.5*(Bc2c1-Bc1s1);
       AApcos[2]=Bp1cn;
       AApsin[2]=Bp2cn;
       AAccos[2]=Bc1cn;
       AAcsin[2]=Bc2cn;
       AApcos[3]=0.5*(Bp1c1-Bp2s1);
       AApsin[3]=0.5*(Bp2c1+Bp1s1);
       AAccos[3]=0.5*(Bc1c1-Bc2s1);
       AAcsin[3]=0.5*(Bc2c1+Bc1s1);
       AApcos[4]=0.5*(Bp1c2-Bp2s2);
       AApsin[4]=0.5*(Bp2c2+Bp1s2);
       AAccos[4]=0.5*(Bc1c2-Bc2s2);
       AAcsin[4]=0.5*(Bc2c2+Bc1s2);
       int ki;

     std::complex<double> test;
      //std::ofstream fout09("Data/Scratch.dat");
     for (int i=0; i<Nph; i++){
        T = tm_ph[i];
        tim[i] = T;
        xi = tm_ph[i] - DM*cos(orbOm*tm_ph[i]-S.phS);
        ind = 0;
        phi[i] = phiev[i];
        gam[i] = gamev[i];
        alph[i] = alpev[i];
        // loops over harmonics
        for (int j=0;j<5;j++) {
        	  nn=(double)(j+1);
        	  for(int jj=0; jj<5; jj++){
        	     mm = (double)jj-2.;
                     dom = nn*phiddotev[i] + 2.*gamddotev[i]  + mm*alpddotev[i];
        	     om = nn*LISAWP_TWOPI*nuev[i] + 2.*gamdotev[i]  + mm*alpdotev[i];
                     faza = (nn*phiev[i] + 2.*gamev[i]  + mm*alpev[i]);
                     ec = eccev[i];
                     ec2=ec*ec;
                     ec3=ec2*ec;
                     ec4=ec3*ec;
                     ec5=ec4*ec;
                     ec6=ec5*ec;
                     ec7=ec6*ec;
                     amp = Ampev[i];
                     switch (j+1) {
               			   case 1:
               				   hamp =3.*ec-1.625*ec3;
               				   break;
               			   case 2:
               				   hamp =-4.+10.*ec2-5.75*ec4;
               				   break;
               			   case 3:
               				   hamp =-9.*ec+21.375*ec3-15.046875*ec5;
               				   break;
               			   case 4:
               				   hamp =-16.*ec2+40.*ec4-101.*ec6/3.;
               				   break;
               			   case 5:
               				   hamp =-625.*ec3/24.+26875.*ec5/384.-210625.*ec7/3072.;
               				   break;
               	     }
                     cFp = Xp[ind][i];
                     cFc = Xc[ind][i];

                     Apc = hamp*AApcos[harmms[jj]+2];
         	     Aps = hamp*AApsin[harmms[jj]+2]; // should be "-" for strain
         	     Acc = 2.*hamp*AAccos[harmms[jj]+2];
         	     Acs = 2.*hamp*AAcsin[harmms[jj]+2];

                     phase[ind][i] =  faza - om*xi + LISAWP_PI*0.25; // NOTE that this is SPA phase
                     frqs[ind][i] = om;

                     fact = 0.5*amp*sqrt(LISAWP_TWOPI/dom);

                     Ampl = fact*( cFp*(Apc - img*Aps) + cFc*(Acc - img*Acs) );
                     X_r[ind][i] = Ampl.real();
                     X_i[ind][i] = Ampl.imag();


                     cFp = Yp[ind][i];
                     cFc = Yc[ind][i];
                     Ampl = fact*( cFp*(Apc - img*Aps) + cFc*(Acc - img*Acs) );
                     Y_r[ind][i] = Ampl.real();
                     Y_i[ind][i] = Ampl.imag();

                     cFp = Zp[ind][i];
                     cFc = Zc[ind][i];
                     Ampl = fact*( cFp*(Apc - img*Aps) + cFc*(Acc - img*Acs) );
                     Z_r[ind][i] = Ampl.real();
                     Z_i[ind][i] = Ampl.imag();

                     ind += 1;


                  }// jj loop
              } //j loop
     } // loop over time

}


double FreqAK_RA::Sinc(double y){
	double z;
	if(y != 0.0){
		z = sin(y)/y;
	}else{
		z = 1.0;
	}
	return(z);
}
