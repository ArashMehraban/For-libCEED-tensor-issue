function y = tensorloop(ne,dof,P,Q,Rf,Sf,Tf,x)
  
   x = permute(reshape(x,P*P*P,[],ne),[2,1]);

   R = zeros(Q,P);
   S = zeros(Q,P);
   T = zeros(Q,P);
    
   for i=1:Q
       for j=1:P
            R(i,j) = Rf((i-1)*P+j);
            S(i,j) = Sf((i-1)*P+j);
            T(i,j) = Tf((i-1)*P+j);
        end
   end
   
   u = zeros(dof,Q*P*P,ne); %holder variable
   for i=1:P
        for l=1:dof
            for a=1:Q
                for jk=1:P*P
                    for e=1:ne
                        u(l, (a-1)*P*P+jk,e) = u(l, (a-1)*P*P+jk,e) + R(a,i) * x(l,(i-1)*P*P+jk,e);
                    end
                end
             end
         end
    end
 
     v = zeros(dof,Q*Q*P,ne); %holder variable
     % v[l,a,b,k] = S[b,j] u[l,a,j,k]
        for l=1:dof
          for a=1:Q
            for k=1:P
              for j=1:P
                for b=1:Q
                  for e=1:ne
                      v(l, (((a-1)*Q+b)-1)*P+k,e) = v(l, (((a-1)*Q+b)-1)*P+k,e) + S(b,j) * u(l, (((a-1)*P+j)-1)*P+k,e);
                  end
                end
              end
            end
          end
        end
        
        
        y = zeros(dof,Q*Q*Q,ne); %output
        % y[l,a,b,c] = T[c,k] v[l,a,b,k]
        for l=1:dof
          for ab=1:Q*Q
            for k=1:P
              for c=1:Q
                for e=1:ne
                    y(l, (ab-1)*Q+c,e) = y(l, (ab-1)*Q+c,e) + T(c,k) * v(l, (ab-1)*P+k,e);
                end
              end
            end
          end
        end      
end

