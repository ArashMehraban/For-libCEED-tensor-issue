/* IMPORTANT NOTE: compile this code with: 
   gcc -std=c99 tensor.c -o tensor -lm
   
  if -lm is not there, you will get an error regarding cos() function 
*/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

typedef enum {TENSOR_EVAL,TENSOR_TRANSPOSE} TensorMode;

int Tensor(int ne,int dof,int P,int Q,double Rf[],double Sf[],double Tf[],TensorMode tmode, double xx[],double *yy)
{

  if (tmode == TENSOR_TRANSPOSE) {int tmp = Q; Q = P; P = tmp;}
  {
    double R[Q][P],S[Q][P],T[Q][P];
    const double (*restrict x)[P*P*P][ne] = (const double(*)[P*P*P][ne])xx;
    double       (*restrict y)[P*P*P][ne] =       (double(*)[Q*Q*Q][ne])yy;
    double u[dof][Q*P*P][ne],v[dof][Q*Q*P][ne];

    for (int i=0; i<Q; i++) {
      for (int j=0; j<P; j++) {
        R[i][j] = tmode == TENSOR_EVAL ? Rf[i*P+j] : Rf[j*Q+i];
        S[i][j] = tmode == TENSOR_EVAL ? Sf[i*P+j] : Sf[j*Q+i];
        T[i][j] = tmode == TENSOR_EVAL ? Tf[i*P+j] : Tf[j*Q+i];
      }
    }
     
    // u[l,a,j,k] = R[a,i] x[l,i,j,k]
    memset(u, 0, dof*Q*P*P*ne*sizeof(double));
    for (int i=0; i<P; i++) {
      for (int l=0; l<dof; l++) {
        for (int a=0; a<Q; a++) {
          for (int jk=0; jk<P*P; jk++) {
            for (int e=0; e<ne; e++){  u[l][a*P*P+jk][e] += R[a][i] * x[l][i*P*P+jk][e];     
//                printf("R[%d][%d] * x[%d][%d][%d]   ", a,i,l,i*P*P+jk,e);
     	    }
//            printf("\n");
          }
       }
     }
    }
    // v[l,a,b,k] = S[b,j] u[l,a,j,k]
    memset(v, 0, dof*Q*Q*P*ne*sizeof(double));
    for (int l=0; l<dof; l++) {
      for (int a=0; a<Q; a++) {
        for (int k=0; k<P; k++) {
          for (int j=0; j<P; j++) {
            for (int b=0; b<Q; b++) {
              for (int e=0; e<ne; e++)v[l][(a*Q+b)*P+k][e] += S[b][j] * u[l][(a*P+j)*P+k][e];
            }
          }
        }
      }
    }

    // y[l,a,b,c] = T[c,k] v[l,a,b,k]
    for (int l=0; l<dof; l++) {
      for (int ab=0; ab<Q*Q; ab++) {
        for (int k=0; k<P; k++) {
          for (int c=0; c<Q; c++) {
            for (int e=0; e<ne; e++) y[l][ab*Q+c][e] += T[c][k] * v[l][ab*P+k][e];
          }
        }
      }
    }
  }
return 0;
}

// Gauss quadrature and weight function (also known as Gauss-Legendre)
int GaussQuadrature(int Q, double *qref1d, double *qweight1d) {
  // Allocate
  double P0, P1, P2, dP2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  for (int i = 0; i <= Q/2; i++) {
    // Guess
    xi = cos(PI*(double)(2*i+1)/((double)(2*Q)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j <= Q; j++) {
      P2 = (((double)(2*j-1))*xi*P1-((double)(j-1))*P0)/((double)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi*P2 - P0)*(double)Q/(xi*xi-1.0);
    xi = xi-P2/dP2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(P2)>1e-15; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j <= Q; j++) {
        P2 = (((double)(2*j-1))*xi*P1-((double)(j-1))*P0)/((double)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(double)Q/(xi*xi-1.0);
      xi = xi-P2/dP2;
    }
    // Save xi, wi
    wi = 2.0/((1.0-xi*xi)*dP2*dP2);
    qweight1d[i] = wi;
    qweight1d[Q-1-i] = wi;
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}

//Gauss-Legendre-Lobatto quadrature and weight function
int LobattoQuadrature(int Q, double *qref1d, double *qweight1d) {
  // Allocate
  double P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  // Set endpoints
  wi = 2.0/((double)(Q*(Q-1)));
  if (qweight1d) {
    qweight1d[0] = wi;
    qweight1d[Q-1] = wi;
  }
  qref1d[0] = -1.0;
  qref1d[Q-1] = 1.0;
  // Interior
  for (int i = 1; i <= (Q-1)/2; i++) {
    // Guess
    xi = cos(PI*(double)(i)/(double)(Q-1));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j < Q; j++) {
      P2 = (((double)(2*j-1))*xi*P1-((double)(j-1))*P0)/((double)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton step
    dP2 = (xi*P2 - P0)*(double)Q/(xi*xi-1.0);
    d2P2 = (2*xi*dP2 - (double)(Q*(Q-1))*P2)/(1.0-xi*xi);
    xi = xi-dP2/d2P2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(dP2)>1e-15; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j < Q; j++) {
        P2 = (((double)(2*j-1))*xi*P1-((double)(j-1))*P0)/((double)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(double)Q/(xi*xi-1.0);
      d2P2 = (2*xi*dP2 - (double)(Q*(Q-1))*P2)/(1.0-xi*xi);
      xi = xi-dP2/d2P2;
    }
    // Save xi, wi
    wi = 2.0/(((double)(Q*(Q-1)))*P2*P2);
    if (qweight1d) {
      qweight1d[i] = wi;
      qweight1d[Q-1-i] = wi;
    }
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}


int FEBasisEval(int P, int Q, double nodes[], double qref1d[],double *interp1d,double *grad1d){

  int i,j,k;
  double c1, c2, c3, c4, dx;
  // Build B, D matrix
  // Fornberg, 1998
  for (i = 0; i  < Q; i++) {
    c1 = 1.0;
    c3 = nodes[0] - qref1d[i];
    interp1d[i*P+0] = 1.0;
    for (j = 1; j < P; j++) {
      c2 = 1.0;
      c4 = c3;
      c3 = nodes[j] - qref1d[i];
      for (k = 0; k < j; k++) {
        dx = nodes[j] - nodes[k];
        c2 *= dx;
        if (k == j - 1) {
          grad1d[i*P + j] = c1*(interp1d[i*P + k] - c4*grad1d[i*P + k]) / c2;
          interp1d[i*P + j] = - c1*c4*interp1d[i*P + k] / c2;
        }
        grad1d[i*P + k] = (c3*grad1d[i*P + k] - interp1d[i*P + k]) / dx;
        interp1d[i*P + k] = c3*interp1d[i*P + k] / dx;
      }
      c1 = c2;
    }
  }
 return 0;
}

int main(){
 
 int P = 2;
 int Q = 2;
 int dof = 1;
 int ne = 1;
 int x_sz = P*P*P*dof*ne;
 int y_sz = Q*Q*Q*dof*ne;

 printf("columns of Identity matrix as input to Tensor():\n\n");
 double Iden[x_sz][x_sz];
 for(int i=0; i<x_sz;i++){
    for(int j=0; j<x_sz;j++){
       if(i == j)
          Iden[i][j] = 1.0;
       else
          Iden[i][j] = 0.0;
    }   
 }

 for(int i=0; i<x_sz; i++){
    for(int j=0; j<x_sz; j++){
      printf("%g ",Iden[i][j]);
    }   
    printf("\n");
}

printf("\n");
printf("Storage for returns from Tensor() column-wise:\n\n");

double Y[y_sz][y_sz];
for(int i=0; i<y_sz;i++)
   for(int j=0; j<y_sz;j++)
       Y[i][j] = 0.0;

for(int i=0; i<y_sz; i++){
   for(int j=0; j<y_sz; j++){
        printf("%g ",Y[i][j]);
   }   
   printf("\n");
  }
 
 printf("\n");
 
 double *qref1d = (double*)calloc(Q, sizeof(double));
 double *w1d = (double*)calloc(Q, sizeof(double));
 GaussQuadrature(Q, qref1d,w1d);
 double *nodes = (double*)calloc(P, sizeof(double));
 LobattoQuadrature(P, nodes, NULL);
 double *interp1d = (double*)calloc(P*Q, sizeof(double));
 double *grad1d = (double*)calloc(P*Q, sizeof(double));
 FEBasisEval(P, Q, nodes, qref1d, interp1d, grad1d);

 double V[x_sz];
 double tmpY[y_sz];
 int k = 0;
 for(int j=0; j<x_sz; j++){
    for(int i=0; i<y_sz; i++){
        V[i] = Iden[i][j];
    }   
    memset(tmpY, 0, y_sz*sizeof(double));
	//                  D        B           B         
    Tensor(ne,dof,P,Q, grad1d, interp1d, interp1d,TENSOR_EVAL, V,tmpY);
	for(k=0; k<y_sz;k++)
	   Y[k][j] = tmpY[k];
 }

 printf("\n");
 printf("D, B, B:\n\n");
 for(int i=0; i<y_sz; i++){
   for(int j=0; j<y_sz; j++){
       printf("%g ",Y[i][j]);
   }
   printf("\n");
 }

 printf("\nThe above is D2, not D0\n");

 
 free(qref1d);
 free(w1d);
 free(nodes);
 free(interp1d);
 free(grad1d);

 return 0;
}

