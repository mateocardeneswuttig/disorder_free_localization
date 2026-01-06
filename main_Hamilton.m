function [ ] = main1( )
tic
%% Microscopic parameters
J = 1;                 % Hopping amplitude for fermions
V = 0;                 % Interaction strength (currently unused)
%omega_value = [pi/2,pi*(sqrt(2)/2),pi,sqrt(2)*pi,2*pi];
omega_value = [pi/2];  % Phonon frequencies to consider
omegaJ_value = [1/2,sqrt(2)/2,1,1.414,2]; % Not currently used
alpha = sqrt(2);       % Coherent state displacement parameter
g_values = 0.25;       % Electron-phonon coupling constants
ave = [0.5767, 0.5688, 0.5284, 0.2295, 0.1586]; % Average values for plotting reference lines
colors = {' (0.1729, 0.6275, 0.1729)', '(0.5020, 0.0000, 0.5020)', '(1.0000, 0.4980, 0.0549)', '(0.8392, 0.1529, 0.1569)','(0.1216, 0.4667, 0.7059)'}; % Colors for plotting

%% Build time vector
myTimes = 0:2:200; % Time points for simulation

%% Choose number representation considering half-filling
Nmax = 1;            % Maximum local phonon occupation
L = 4;               % Number of lattice sites
U1sector = L;        % Number of fermions (half-filling)
HilbertSpaceSize = nchoosek(2*L,U1sector) * ((Nmax+1)^L);  % Total Hilbert space size
I = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
    ones(1,HilbertSpaceSize),HilbertSpaceSize,HilbertSpaceSize); % Identity matrix

% Preallocate cell arrays for operators
tunneling_up = cell(1,L);        % Spin-up fermion hopping operators
tunneling_down = cell(1,L);      % Spin-down fermion hopping operators
interaction = cell(1,L);         % Interaction operators (unused)
phononCreator = cell(1,L);       % Phonon creation operators
phononAnnihilator = cell(1,L);   % Phonon annihilation operators
matterOccupation_up = cell(1,L); % Spin-up occupation number operators
matterOccupation_down = cell(1,L); % Spin-down occupation number operators
phononOccupation = cell(1,L);    % Phonon number operators

%% Build local operators for each site
for m = 1:L % Loop over lattice sites
    m
    % Initialize lists for sparse matrix construction
    tunnelingListAbscissa_up = [];
    tunnelingListOrdinates_up = [];
    tunnelingListAbscissa_down = [];
    tunnelingListOrdinates_down = [];
    phononCreatorAbscissa = [];
    phononCreatorOrdinates = [];
    sign_up = [];
    sign_down = [];
    phononCreatorVals = [];
    phononAnnihilatorAbscissa = [];
    phononAnnihilatorOrdinates = [];
    phononAnnihilatorVals = [];
    
    % Ladder vector for phonon creation at site m
    ladderVec = zeros(1,3*L); 
    ladderVec(3*m) = 1;
    
    % Vectors for fermion hopping
    vec = zeros(1,3*L); 
    vec(3*m-2) = 1;                     % Spin-up hopping from site m
    vec(mod(3*m,3*L)+1) = -1;           % Spin-up hopping to neighboring site
    vec_down = zeros(1,3*L); 
    vec_down(3*m-1) = 1;                 % Spin-down hopping from site m
    vec_down(mod(3*m,3*L)+2) = -1;       % Spin-down hopping to neighboring site

    % Initialize occupation number arrays
    matterOccupationVal = zeros(1,HilbertSpaceSize);
    matterOccupationVal_up1 = zeros(1,HilbertSpaceSize); % Neighboring site occupation for special phase
    matterOccupationVal_down = zeros(1,HilbertSpaceSize);
    phononOccupationVal = zeros(1,HilbertSpaceSize);
    jump_counter = 0;
    total = 0;
    
    %% Loop over all Hilbert space basis states
    for cnt = 1:HilbertSpaceSize
        occupat = 0; 
        psi = label2fock(cnt,L,U1sector,Nmax); % Convert label to Fock state
        % Count total spin-up occupation
        for i = 1:L
            occupat = occupat + psi(3*i-2);
        end
        matterOccupationVal_up(cnt) = psi(3*m-2); % Spin-up occupation at site m
        matterOccupationVal_up1(cnt) = psi(mod(3*m,3*L)+1); % Neighboring site
        matterOccupationVal_down(cnt) = psi(3*m-1); % Spin-down occupation at site m
        phononOccupationVal(cnt) = psi(3*m);       % Phonon occupation at site m

        %% Phonon creation operator
        phi = psi + ladderVec;
        if phi(3*m) <= Nmax && phi(3*m) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            if tnc > HilbertSpaceSize
                keyboard % Debugging
            end
            phononCreatorAbscissa = [phononCreatorAbscissa cnt];
            phononCreatorOrdinates = [phononCreatorOrdinates tnc];
            phononCreatorVals = [phononCreatorVals sqrt(psi(3*m)+1)];
        end

        %% Phonon annihilation operator
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

        %% Spin-up fermion hopping
        phi = psi + vec;
        if max(phi(1:3:end)) <= 1 && max(phi(2:3:end)) <= 1 && max(phi(3:3:end)) <= Nmax && min(phi) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            tunnelingListAbscissa_up = [tunnelingListAbscissa_up cnt tnc];
            tunnelingListOrdinates_up = [tunnelingListOrdinates_up tnc cnt];
            if m == L
                a = -sign(m-L+0.5) * (-1)^occupat; % Apply phase factor at boundary
            else
                a = -sign(m-L+0.5);
            end
            sign_up = [sign_up a a];
            total = total - 1;
        end

        %% Spin-down fermion hopping
        phi = psi + vec_down;
        if max(phi(2:3:end)) <= 1 && max(phi(1:3:end)) <= 1 && max(phi(3:3:end)) <= Nmax && min(phi) >= 0
            tnc = fock2label(phi,L,U1sector,Nmax);
            tunnelingListAbscissa_down = [tunnelingListAbscissa_down cnt tnc];
            tunnelingListOrdinates_down = [tunnelingListOrdinates_down tnc cnt];
            if m == L
                a = -sign(m-L+0.5) * (-1)^occupat;
            else
                a = -sign(m-L+0.5);
            end
            sign_down = [sign_down a a];
        end
    end

    %% Convert occupation and operator lists to sparse matrices
    matterOccupation_up{m} = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
        matterOccupationVal,HilbertSpaceSize,HilbertSpaceSize);
    
    matterOccupation_down{m} = sparse(1:HilbertSpaceSize,1:HilbertSpaceSize,...
        matterOccupationVal_down,HilbertSpaceSize,HilbertSpaceSize);

    phononCreator{m} = sparse(phononCreatorOrdinates,phononCreatorAbscissa,...
        phononCreatorVals,HilbertSpaceSize,HilbertSpaceSize);

    phononAnnihilator{m} = sparse(phononAnnihilatorOrdinates,phononAnnihilatorAbscissa,...
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
H_e_a = NaN * ones(size(omega_value));
H_ph_a = NaN * ones(size(omega_value));
H_e_ph_a = NaN * ones(size(omega_value));
H_e_0 = NaN * ones(size(omega_value));
Ph_a = NaN * ones(size(omega_value));
matterOccupation = NaN * ones(size(omega_value));

%% Loop over different phonon frequencies
for omega_index = 1:length(omega_value)
    omega = omega_value(omega_index);
    omega
    color = colors{omega_index};

    %% Initialize Hamiltonians and operators
    H_e = 0 * I;      % Electronic part
    H_ph = 0 * I;     % Phonon part
    H_e_ph = 0 * I;   % Electron-phonon coupling
    H = 0 * I;        % Total Hamiltonian
    Matter = 0 * I;   % Total matter occupation operator
    Ph = 0 * I;       % Total phonon occupation operator

    for m = 1:L
        H_e = H_e - J * tunneling_up{m} - J * tunneling_down{m}; % Hopping
        H_ph = H_ph + omega * (phononCreator{m} * phononAnnihilator{m}); % Phonons
        H_e_ph = H_e_ph + g_values * (matterOccupation_up{m} + matterOccupation_down{m} - I) * (phononAnnihilator{m} + phononCreator{m})^2; % Coupling
        Matter = Matter + matterOccupation_up{m};
        Ph = Ph + phononOccupation{m};
    end
    H = H_e + H_ph + H_e_ph; % Total Hamiltonian
    d = eigs(H, HilbertSpaceSize); % Eigenvalues for diagnostics
    d

    %% Prepare initial state
    % Spin-up and spin-down configurations and coefficients
    up_states = {[1,0,0,1,0,0,0,0,0,0,0,0],  
                 [1,0,0,0,0,0,1,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0,0,1,0,0],   
                 [0,0,0,1,0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,1,0,0]};   
    up_coeffs = [1-1i, 2, 1+1i, 1+1i, 2i, 1i-1]; 

    down_states = {[0,1,0,0,1,0,0,0,0,0,0,0],  
                   [0,1,0,0,0,0,0,1,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,1,0,0,1,0,0,0,0],  
                   [0,0,0,0,1,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,1,0,0,1,0]};  
    down_coeffs = [1-1i, 2, 1+1i, 1+1i, 2i, 1i-1];

    phononmax = [0,0,Nmax,0,0,Nmax,0,0,Nmax,0,0,Nmax];

    %% Construct initial superposition state
    psi0 = 0*I(:,1);
    psi_test = 0*I(:,1);
    for k = 1:6
        for n = 1:6
            psi1 = 0*I(:,1);
            statefirst = up_states{k} + down_states{n};
            statelast = up_states{k} + down_states{n} + phononmax;
            beginInd = fock2label(statefirst,L,U1sector,Nmax);
            endInd = fock2label(statelast,L,U1sector,Nmax);

            for cnt = beginInd:endInd 
                coeff = exp(-L*(abs(alpha)^2)/2);
                for m = 1:L
                    locOcc = I(:,cnt)' * phononOccupation{m} * I(:,cnt);
                    coeff = coeff * (alpha^(locOcc)) / sqrt(factorial(locOcc));
                end
                if cnt == beginInd
                    psi1 = coeff * I(:,cnt);
                else
                    psi1 = psi1 + coeff * I(:,cnt);
                end
            end
            psi0 = psi0 + up_coeffs(k) * down_coeffs(n) * psi1 / 16;
        end
    end
    psi0 = psi0 / norm(psi0); % Normalize initial state

    %% Time evolution
    He_t = NaN * ones(size(myTimes));
    dt = myTimes(2) - myTimes(1);

    for tInd = 1:numel(myTimes)
        t = myTimes(tInd);
        disp(['t=' num2str(t)])
        if tInd == 1
            psi = psi0;
        else
            psi = expv(dt, -1i * H, psi); % Time evolution via matrix exponential
        end
        He_t(tInd) = real(psi' * H_e * psi) - real(psi0' * H_ph * psi0);
        He_t(tInd)/4
        H_ph0 = real(psi0' * H_e * psi0);
    end

    H_e_a(omega_index) = mean(He_t(80:end));
    matterOccupation(omega_index) = psi' * Matter * psi;

    %% Plot time evolution
    plot(myTimes, He_t/4, 'color', color, 'linewidth', 2);
    plot([myTimes(1), myTimes(end)], [ave(omega_index), ave(omega_index)], '--', 'color', color, 'linewidth', 2);
end

hold off;
yticks([-0.5,0,0.5,1]);
ax = gca;
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
legend('L=4');
saveas(gcf, 'Repro_L4_He_final_100_time.svg');
toc
end


