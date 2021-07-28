vals = StringSplit[InputString[]]

p = ToExpression[vals[[1]]];
q = ToExpression[vals[[2]]];
Print[p];
Print[q];

\[Phi] = 2 Pi p/q;

a1 = {1, 0};
a2 = {0, 1};
b1 = {1, 0};
b2 = {0, 1};
ta1 = 1/p*a1;
ta2 = q a2;
tb1 = p b1;
tb2 = 1/q*b2;
\[CapitalOmega] = Abs[ta1[[1]]*ta2[[2]] - ta2[[1]]*ta1[[2]]];
z1 = 1/Sqrt[\[CapitalOmega]]*(ta1[[1]] + I*ta1[[2]]);
z2 = 1/Sqrt[\[CapitalOmega]]*(ta2[[1]] + I*ta2[[2]]);

numLandau = 100;
\[Gamma]q1 = 2*Pi*b1.ta1*Conjugate[z2] - 2*Pi*b1.ta2*Conjugate[z1];
\[Gamma]q2 = 2*Pi*b2.ta1*Conjugate[z2] - 2*Pi*b2.ta2*Conjugate[z1];

getLaguerreFast = 
  Compile[{{\[Gamma]q, _Complex}, {m, _Integer}, {n, _Integer}}, 
   N[Exp[-\[Gamma]q*Conjugate[\[Gamma]q]/(8 Pi)]*
     Exp[1/2 (LogGamma[Min[n, m] + 1] - LogGamma[Max[n, m] + 1])]*
     If[n >= m, (I*\[Gamma]q/Sqrt[4 Pi])^(n - m), (I*
         Conjugate[\[Gamma]q]/Sqrt[4 Pi])^(m - n)] LaguerreL[
      Min[m, n], Abs[n - m], \[Gamma]q*Conjugate[\[Gamma]q]/(4 Pi)]]];
      
hqp1 = Table[
   getLaguerreFast[\[Gamma]q1, m, n], {m, 0, numLandau - 1}, {n, 0, 
    numLandau - 1}];
hqp2 = Table[
   getLaguerreFast[\[Gamma]q2, m, n], {m, 0, numLandau - 1}, {n, 0, 
    numLandau - 1}];
    
Export[StringJoin["formFactor1_", ToString[p], "_", ToString[q], 
  ".csv"], hqp1, "CSV"]
  
Export[StringJoin["formFactor2_", ToString[p], "_", ToString[q], 
  ".csv"], hqp2, "CSV"]