/* Extracted verbatim from the Clean -> Estimate dashboard (artifact version 1792415312-3262), lines 1158-1195: its OLS engine. */
function gammln(xx){const cof=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];let x=xx,y=xx,tmp=x+5.5;tmp-=(x+0.5)*Math.log(tmp);let ser=1.000000000190015;for(let j=0;j<6;j++){y+=1;ser+=cof[j]/y;}return -tmp+Math.log(2.5066282746310005*ser/x);}
function betacf(a,b,x){const FPMIN=1e-30,EPS=3e-12,MAXIT=200;let qab=a+b,qap=a+1,qam=a-1,c=1,d=1-qab*x/qap;if(Math.abs(d)<FPMIN)d=FPMIN;d=1/d;let h=d;for(let m=1;m<=MAXIT;m++){let m2=2*m,aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;d=1/d;h*=d*c;aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d;if(Math.abs(d)<FPMIN)d=FPMIN;c=1+aa/c;if(Math.abs(c)<FPMIN)c=FPMIN;d=1/d;let del=d*c;h*=del;if(Math.abs(del-1)<EPS)break;}return h;}
function betai(a,b,x){if(x<=0)return 0;if(x>=1)return 1;const bt=Math.exp(gammln(a+b)-gammln(a)-gammln(b)+a*Math.log(x)+b*Math.log(1-x));if(x<(a+1)/(a+b+2))return bt*betacf(a,b,x)/a;return 1-bt*betacf(b,a,1-x)/b;}
function tTwoTailP(t,df){if(df<=0||!isFinite(t))return NaN;return betai(df/2,0.5,df/(df+t*t));}
function fUpperP(f,df1,df2){if(f<=0)return 1;return betai(df2/2,df1/2,df2/(df2+df1*f));}
function tCrit(df,alpha){ // two-sided critical t for given alpha via bisection
  let lo=0,hi=100;for(let i=0;i<80;i++){const mid=(lo+hi)/2;(tTwoTailP(mid,df)>alpha)?(lo=mid):(hi=mid);}return (lo+hi)/2;}

/* ===== linear algebra ===== */
function matT(A){const r=A.length,c=A[0].length,B=[];for(let j=0;j<c;j++){B[j]=[];for(let i=0;i<r;i++)B[j][i]=A[i][j];}return B;}
function matMul(A,B){const r=A.length,n=B.length,c=B[0].length,C=[];for(let i=0;i<r;i++){C[i]=new Array(c).fill(0);for(let k=0;k<n;k++){const a=A[i][k];if(a===0)continue;for(let j=0;j<c;j++)C[i][j]+=a*B[k][j];}}return C;}
function matVec(A,v){const r=A.length,n=v.length,o=new Array(r).fill(0);for(let i=0;i<r;i++){let s=0;for(let k=0;k<n;k++)s+=A[i][k]*v[k];o[i]=s;}return o;}
function matInv(M){const n=M.length,A=M.map((row,i)=>row.concat(Array.from({length:n},(_,j)=>(i===j?1:0))));for(let col=0;col<n;col++){let piv=col,max=Math.abs(A[col][col]);for(let r=col+1;r<n;r++){if(Math.abs(A[r][col])>max){max=Math.abs(A[r][col]);piv=r;}}if(max<1e-14)throw new Error("singular");if(piv!==col){const t=A[piv];A[piv]=A[col];A[col]=t;}const pv=A[col][col];for(let j=0;j<2*n;j++)A[col][j]/=pv;for(let r=0;r<n;r++){if(r===col)continue;const f=A[r][col];if(f===0)continue;for(let j=0;j<2*n;j++)A[r][j]-=f*A[col][j];}}return A.map(row=>row.slice(n));}

/* ===== OLS ===== */
function ols(X,y,seType,hasConst){
  if(hasConst===undefined)hasConst=true;
  const n=X.length,k=X[0].length;
  if(k===0)throw new Error("few-obs");
  if(n<=k)throw new Error("few-obs");
  const Xt=matT(X),XtX=matMul(Xt,X),XtXinv=matInv(XtX),beta=matVec(XtXinv,matVec(Xt,y));
  const yhat=matVec(X,beta),resid=y.map((v,i)=>v-yhat[i]);
  const rss=resid.reduce((s,e)=>s+e*e,0);
  let tss,dfM,dfT;
  if(hasConst){ const ybar=y.reduce((s,v)=>s+v,0)/n; tss=y.reduce((s,v)=>s+(v-ybar)*(v-ybar),0); dfM=k-1; dfT=n-1; }
  else { tss=y.reduce((s,v)=>s+v*v,0); dfM=k; dfT=n; }
  const dfR=n-k, r2=1-rss/tss, r2adj=1-(1-r2)*dfT/dfR, s2=rss/dfR;
  let cov;
  if(seType==='robust'){
    const meat=Array.from({length:k},()=>new Array(k).fill(0));
    for(let i=0;i<n;i++){const xi=X[i],e2=resid[i]*resid[i];for(let a=0;a<k;a++){const xa=xi[a]*e2;for(let b=0;b<k;b++)meat[a][b]+=xa*xi[b];}}
    cov=matMul(matMul(XtXinv,meat),XtXinv);const c=n/(n-k);for(let a=0;a<k;a++)for(let b=0;b<k;b++)cov[a][b]*=c;
  } else { cov=XtXinv.map(row=>row.map(v=>v*s2)); }
  const se=cov.map((r,i)=>Math.sqrt(r[i])),tv=beta.map((b,i)=>b/se[i]),pv=tv.map(t=>tTwoTailP(t,dfR));
  const fstat=dfM>0?(r2/dfM)/((1-r2)/dfR):NaN,fp=dfM>0?fUpperP(fstat,dfM,dfR):NaN;
  const ess=tss-rss, msModel=dfM>0?ess/dfM:NaN, msResid=s2, msTotal=tss/dfT, rootmse=Math.sqrt(s2);
  return {beta,se,tv,pv,r2,r2adj,n,k,dfR,dfM,dfT,fstat,fp,rss,tss,ess,msModel,msResid,msTotal,rootmse,cov,seType,resid,hasConst};
}
module.exports = { ols };
