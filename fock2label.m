function [m] = fock2label(psi, L, filling, Nmax)
% Converts a Fock state vector `psi` to a Hilbert space index (label)
% The Fock state `psi` is organized as [up1, down1, phonon1, up2, down2, phonon2, ...]
%
% Inputs:
%   psi     - 1x(3*L) vector representing occupation numbers
%             (up, down, phonon for each site)
%   L       - Number of lattice sites
%   filling - Total number of fermions (sum of all spins)
%   Nmax    - Maximum allowed phonon occupation per site
%
% Output:
%   m       - Hilbert space label corresponding to the Fock state `psi`

%% Split fermion (two spins) and phonon components
psi_matter_up = psi(1:3:end);      % Spin-up occupation numbers
psi_matter_down = psi(2:3:end);    % Spin-down occupation numbers
psi_phonons = psi(3:3:end);        % Phonon occupation numbers
psi_phononsString = NaN*ones(1,L); % Placeholder for phonon digits in base (Nmax+1)

%% Flatten spin occupations into a single vector of length 2*L
psi_matter_flat = [psi_matter_up, psi_matter_down]; 

%% Check that the total fermion number matches the filling
if sum(psi_matter_flat) ~= filling
    disp('Error: input state is not in the N-particle sector...')
    m = -1;  % Return invalid label
else
    %% Encode fermion part using combinatorial labeling
    n = filling; % Remaining number of fermions to place
    m = 1;       % Start label at 1

    % Loop over all spin orbitals (total 2*L)
    for s = 1:2*L
        if n == 0
            break % All fermions placed
        elseif psi_matter_flat(s) == 0
            % If orbital is empty, add the number of states where remaining fermions
            % could be placed in the remaining orbitals
            m = m + nchoosek(2*L - s, n-1);
        else
            % If orbital is occupied, reduce remaining fermions to place
            n = n - 1;
        end
    end

    %% Encode phonon part as a base-(Nmax+1) number
    % Each phonon occupation acts like a digit in base (Nmax+1)
    for s = 1:L
        psi_phononsString(s) = dec2base(psi_phonons(s), Nmax+1, 1);
    end
    
    %% Combine fermion and phonon encodings into final Hilbert space label
    % Fermion part gives the major index, phonon part gives minor index
    m = (m-1) * ((Nmax+1)^L) + base2dec(psi_phononsString, Nmax+1) + 1;
end
end


