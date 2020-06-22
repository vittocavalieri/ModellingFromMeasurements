function label = setTransitionLabel(x)

x_p = x;
x_p(x<0) = 0;
x_n = x;
x_n(x>0) = 0;
[pks_p, locs_p] = findpeaks(x_p);
[pks_n, locs_n] = findpeaks(-x_n);
pks_n = -pks_n;
[locs, ii] = sort([locs_p;locs_n]);
pks = [pks_p; pks_n];
pks = pks(ii);

label = zeros(length(x),1);
k = 1;
j = 1;
for i = 1:length(x)
    if locs(j) == i && j~=length(locs)
        if pks(j)*pks(j+1) < 0
            k = -k;
        end
        j = j + 1;
    end
    label(i) = k;
end

end