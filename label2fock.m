function [ psi ] = label2fock(cnt, L, filling, Nmax)
% Converts a Hilbert space index (label) `cnt` to a Fock state vector `psi`
% The Fock state vector `psi` has length 3*L: [up1, down1, phonon1, up2, down2, phonon2, ...]
%
% Inputs:
%   cnt     - Hilbert space index (1-based)
%   L       - Number of lattice sites
%   filling - Total number of fermions
%   Nmax    - Maximum phonon occupation per site
%
% Output:
%   psi     - 1x(3*L) vector representing the Fock state

%% Split the Hilbert space label into fermion and phonon parts
cnt_matter = ceil(cnt / ((Nmax+1)^L));   % Index of the fermion (matter) block
cnt_phonons = mod(cnt-1, (Nmax+1)^L) + 1; % Offset within phonon block

%% Decode phonon part
% Convert phonon offset to a string in base-(Nmax+1)
psi_phononsString = dec2base(cnt_phonons-1, Nmax+1, L);
psi_phonons = zeros(1,L); % Initialize phonon occupations
for s = 1:L
    psi_phonons(s) = base2dec(psi_phononsString(s), Nmax+1);
end

%% Decode fermion (matter) part
% Initialize 2*L vector for spin orbitals (up/down)
psi_matter = zeros(1, 2*L);
n = filling; % Number of fermions left to place
m = cnt_matter; % Copy of fermion block index

% Loop over all 2*L spin orbitals
for s = 1:2*L
    if n >= 1
        btmp = nchoosek(2*L - s, n-1); % Number of states if this orbital is empty
    else
        break % All fermions placed
    end
    
    if m > btmp
        % This orbital is empty
        m = m - btmp; 
        psi_matter(s) = 0;
    else
        % This orbital is occupied
        n = n - 1;
        psi_matter(s) = 1;
    end
end

%% Combine fermion and phonon parts into a 3*L vector
psi = zeros(1, 3*L);
for i = 1:L
    % Spin-up occupation for site i
    psi(3*i-2) = psi_matter(i); 
    % Spin-down occupation for site i
    psi(3*i-1) = psi_matter(i+L);   
    % Phonon occupation for site i
    psi(3*i) = psi_phonons(i);      
end
end


