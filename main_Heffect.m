function [ ] = main_H( )
tic
%% Microscopic parameters
J = 1;                  % Hopping amplitude for fermions
V = 0;                  % Interaction strength (currently unused)
omega = pi/2;            % Phonon frequency
alpha = sqrt(2);         % Coherent state displacement parameter
g_values =0.25;          % Electron-phonon coupling constants to consider
colors = {'(0.0, 0.4, 0.7)', '(1,0.6,0)','(0.8039,0.10,0.0810)'}; % Colors for plotting different g values

%% Build time vector
myTimes = 0:10:1000;     % Time points for simulation

%% Choose the number representation considering half-filling
Nmax = 1;                % Maximum local phonon occupation
L = 4;                   % Number of lattice sites
U1sector = L;            % Number of fermions (half-filling)
% Hilbert space size: choose L fermions among 2*L sites and combine with phonon states
HilbertSpaceSize = nchoosek(2*L,U1sector)*((Nmax+1)^L); 
% Identity matrix of the Hilbert space
I = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
    ones(1,HilbertSpaceSize),HilbertSpaceSize,HilbertSpaceSize);

% Preallocate cell arrays for operators
tunneling_up = cell(1,L); % Hopping operators for spin-up fermions
tunneling_down = cell(1,L); % Hopping operators for spin-down fermions
interaction = cell(1,L);    % Interaction operators (currently unused)
phononCreator = cell(1,L);  % Phonon creation operators
phononAnnihilator = cell(1,L); % Phonon annihilation operators
matterOccupation_up = cell(1,L); % Occupation number for spin-up fermions
matterOccupation_down = cell(1,L); % Occupation number for spin-down fermions
phononOccupation = cell(1,L);     % Phonon number operators

%% Build local operators for each site
for m = 1:L % Loop over lattice sites
    m
    % Initialize lists to store sparse matrix elements
    tunnelingListAbscissa_up = [];
    tunnelingListOrdinates_up = [];
    tunnelingListAbscissa_down = [];
    tunnelingListOrdinates_down = [];
    phononCreatorOrdinates = [];
    phononCreatorAbscissa = [];
    phononCreatorVals = [];
    phononAnnihilatorAbscissa = [];
    phononAnnihilatorOrdinates = [];
    phononAnnihilatorVals = [];
    sign_up=[];
    sign_down=[];
    
    % Ladder vector for phonon creation at site m
    ladderVec = zeros(1,3*L); 
    ladderVec(3*m) = 1;

    % Vectors for fermion hopping
    vec = zeros(1,3*L); 
    vec(3*m-2) = 1;                         % Spin-up hopping from site m
    vec(mod(3*m,3*L)+1) = -1;               % Spin-up hopping to neighboring site
    vec_down = zeros(1,3*L); 
    vec_down(3*m-1) = 1;                     % Spin-down hopping from site m
    vec_down(mod(3*m,3*L)+2) = -1;           % Spin-down hopping to neighboring site

    % Initialize arrays to store occupation numbers
    matterOccupationVal_up = zeros(1,HilbertSpaceSize);
    matterOccupationVal_down = zeros(1,HilbertSpaceSize);
    phononOccupationVal = zeros(1,HilbertSpaceSize);
    
    %% Loop over all basis states in the Hilbert space
    for cnt = 1:HilbertSpaceSize
        % Convert Hilbert space index to Fock state
        psi = label2fock(cnt,L,U1sector,Nmax);
        % Store fermion occupations
        matterOccupationVal_up(cnt) = psi(3*m-2);
        matterOccupationVal_down(cnt) = psi(3*m-1);
        % Store phonon occupation
        phononOccupationVal(cnt) = psi(3*m);
        
        %% Phonon creation operator: phi = psi + ladderVec
        phi = psi + ladderVec;
        if phi(3*m) <= Nmax && phi(3*m) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax); % Map Fock state back to index
            if tnc > HilbertSpaceSize
                keyboard % Debugging breakpoint
            end
            % Store nonzero matrix elements for sparse matrix
            phononCreatorAbscissa = [phononCreatorAbscissa cnt];
            phononCreatorOrdinates = [phononCreatorOrdinates tnc];
            phononCreatorVals = [phononCreatorVals sqrt(psi(3*m)+1)];
        end

        %% Phonon annihilation operator: phi = psi - ladderVec
        phi = psi - ladderVec;
        if phi(3*m) <= Nmax && phi(3*m) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            if tnc > HilbertSpaceSize
                keyboard
            end
            phononAnnihilatorAbscissa = [phononAnnihilatorAbscissa cnt];
            phononAnnihilatorOrdinates = [phononAnnihilatorOrdinates tnc];
            phononAnnihilatorVals = [phononAnnihilatorVals sqrt(psi(3*m))];
        end
        
        %% Spin-up fermion tunneling
        phi = psi + vec;
        if max(phi(1:3:end)) <= 1 && max(phi(2:3:end)) <= 1 && max(phi(3:3:end)) <= Nmax && min(phi) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            tunnelingListAbscissa_up = [tunnelingListAbscissa_up cnt tnc];
            tunnelingListOrdinates_up = [tunnelingListOrdinates_up tnc cnt];
            sign_up = [sign_up -sign(m-L+0.5) -sign(m-L+0.5)];
        end
        
        %% Spin-down fermion tunneling
        phi = psi + vec_down;
        if max(phi(2:3:end)) <= 1 && max(phi(1:3:end)) <= 1 && max(phi(3:3:end)) <= Nmax && min(phi) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            tunnelingListAbscissa_down = [tunnelingListAbscissa_down cnt tnc];
            tunnelingListOrdinates_down = [tunnelingListOrdinates_down tnc cnt];
            sign_down = [sign_down -sign(m-L+0.5) -sign(m-L+0.5)];
        end
    end

    %% Convert occupation numbers and operators to sparse matrices
    matterOccupation_up{m} = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
        matterOccupationVal_up,HilbertSpaceSize,HilbertSpaceSize);
    
    matterOccupation_down{m} = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
        matterOccupationVal_down,HilbertSpaceSize,HilbertSpaceSize);

    phononCreator{m} = sparse(phononCreatorAbscissa,phononCreatorOrdinates,...
        phononCreatorVals,HilbertSpaceSize,HilbertSpaceSize);

    phononAnnihilator{m} = sparse(phononAnnihilatorAbscissa,phononAnnihilatorOrdinates,...
        phononAnnihilatorVals,HilbertSpaceSize,HilbertSpaceSize);

    phononOccupation{m} = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
        phononOccupationVal,HilbertSpaceSize,HilbertSpaceSize);

    tunneling_up{m} = sparse(tunnelingListAbscissa_up,tunnelingListOrdinates_up,...
        sign_up,HilbertSpaceSize,HilbertSpaceSize);
        
    tunneling_down{m} = sparse(tunnelingListAbscissa_down,tunnelingListOrdinates_down,...
        sign_down,HilbertSpaceSize,HilbertSpaceSize);
end

toc

%% Build Hamiltonians and other operators
figure, hold on;

for g_idx = 1:length(g_values)
    g = g_values(g_idx);
    color = colors{g_idx};
    % Renormalized hopping amplitude and phonon frequency
    J_star = J*exp(-1/2*(g/omega)^2*(alpha^2+2*alpha+1));
    omega_star = omega-(g^2)/omega;

    %% Initialize Hamiltonian and imbalance operator
    H = 0 * I;
    imbOp = 0 * I;
    
    %% Construct the Hamiltonian
    for m = 1:L
        H = H - J_star * tunneling_up{m} - J_star * tunneling_down{m} ...
            + omega_star * phononOccupation{m} ...
            + 2*g * (matterOccupation_up{m}+matterOccupation_down{m} - I) .* (phononOccupation{m}+1/2*I) ...
            - 4*(g^2)/omega*(matterOccupation_up{m}-1/2*I).*(matterOccupation_down{m}-1/2*I).*(phononOccupation{m}+1/2*I);
    end
    
    %% Prepare initial coherent state
    targetMatterConfig = mod(1:L,2);  % Alternating occupation (0,1,0,1...)
    firstState = zeros(1,3*L);
    firstState(1:3:end) = targetMatterConfig;
    firstState(2:3:end) = targetMatterConfig;
    lastState = Nmax * ones(1,3*L);
    lastState(1:3:end) = targetMatterConfig;
    lastState(2:3:end) = targetMatterConfig;
    beginInd = fock2label(firstState,L,U1sector,Nmax);
    endInd = fock2label(lastState,L,U1sector,Nmax);
    
    psi0 = 0; % Initialize the wavefunction
    for cnt = beginInd:endInd 
        coeff = exp(-L*(abs(alpha)^2)/2);
        for m = 1:L
            locOcc = I(:,cnt)' * phononOccupation{m} * I(:,cnt);
            coeff = coeff * (alpha^(locOcc)) / sqrt(factorial(locOcc));
        end
        
        if cnt == beginInd
            psi0 = coeff * I(:,cnt);
        else
            psi0 = psi0 + coeff * I(:,cnt);
        end
    end
    psi0 = psi0 / norm(psi0);  % Normalize initial state
    
    %% Build imbalance operator
    for m = 1:L
        imbOp = imbOp + (2 * psi0' * matterOccupation_up{m} * psi0 - 1) * matterOccupation_up{m} / L;
    end
    
    %% Time evolution
    imb = NaN * ones(size(myTimes));
    dt = myTimes(2) - myTimes(1);
    
    for tInd = 1:numel(myTimes)
        t = myTimes(tInd);
        disp(['t=' num2str(t)])
        if tInd == 1
            psi = psi0;
        else
            psi = expv(dt, -1i * H, psi); % Exponentiate Hamiltonian for time evolution
        end
        
        imb(tInd) = real(psi' * imbOp * psi); % Imbalance expectation value
        occu(tInd) = real(psi' * matterOccupation_up{2} * psi); % Occupation on site 2
    end
    
    %% Plot time evolution
    plot(myTimes, movmean(occu, 10), 'color', color, 'linewidth', 2);
    
    %% Compute and plot average of the last half of the time evolution
    half_idx = round(numel(myTimes) / 2);
    avg_imb = mean(imb(half_idx:end));
    avg_imb
    plot([myTimes(1), myTimes(end)], [avg_imb, avg_imb], '--', 'color', color, 'linewidth', 2);
end

hold off;
xlabel('Time');
ylabel('Imbalance');
ylim([0, 0.8]);
xlim([0,5]);
legend('g=0.35', 'Avg g=0.35', 'g=0.3', 'Avg g=0.3', 'g=0.35', 'Avg g=0.35');
saveas(gcf, 'ImbalancePloteffect5000_0.25_4_2.svg');
grid on;
toc
end

