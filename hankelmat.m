function H = hankelmat(V, L, T)
% HANKELMAT - Compute the hankel matrix of
%   the vector sequence V.
%
%   Syntax
%       hankelmat(V, L)
%       hankelmat(V, L, T)
%
%   Input Arguments
%       V - Column vectors sequence
%       L - Hankel matrix height
%       T - Hankel matrix width
%
%   Output Arguments
%       H - Hankel matrix
%
%   See also HANKEL.

    arguments
        V (:,:) double;
        L (1,1) {mustBeInteger};
        T (1,1) {mustBeInteger} = size(V,2);
    end
    
    m = size(V,1);
    H = zeros([m*L, T-(L-1)]);

    for r = 1:L
        for c = 1:T-(L-1)
            H(m*(r-1)+1:m*r, c) = V(:, r+c-1);
        end
    end
end