function [outk1,outk2,outk3] = distance(x,mu,k1,k2,k3)
n = length(x);
outk1=0;outk2=0;outk3=0;
for i = 1:1:n
    d = sqrt((x(i,1)-mu(1,1))^2+(x(i,2)-mu(1,2))^2);
    if(d<k3)
        outk3= outk3+1;
    end
    if(d<k2)
        outk2= outk2+1;
    end
    if(d<k1)
        outk1= outk1+1;
    end
        
end
