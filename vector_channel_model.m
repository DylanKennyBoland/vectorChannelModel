%% Digital Communications (EEEN40060) Assignment 1:
% This program attempts to simulate an M-ary vector channel.
% M, the number of transmit symbols, is 8, although this can
% be changed at the top of the program. The channel's effect is
% modelled by overlaying the transmit signal (s) with white Gaussian
% noise. The MAP rule is used by the receiver in order to map the
% received signal to one of the M possible transmit ones.
%
% Author: Dylan Boland (Student)

% The boolean variables below are to allow the user to easily control
% whether the SER or BER curves get plotted:
plotSERcurves = true; 
plotBERcurve = true;

% The boolean variable below is set to "true" if the user
% wishes to simulate the bit-to-symbol mapping stage. Setting the
% variable to "false" will skip the bit-to-symbol mapping stage
% and instead just generate a sequence of transmit symbols. Setting this
% to "false" will speed up the running time of the program. However, in
% order to generate the BER curve this variable should be set to "true" as
% well as "plotBERcurve" above:
bits2symbols = true;

% Setup phase of variables and system parameters:
M = 8; % the number of transmit symbols
% The vector below stores the probability that each of the M
% symbols will be transmitted. Since the specification mentions
% that each symbol is equally likely to be transmitted, the
% probability for each is 1/M:
txProbabilities = (1/M)*ones(1, M);
d = 1; % the variable d, as mentioned in the system specification

% We have two orthonormal basis functions ϕ1 and ϕ2. When a given
% waveform s(t) is transmitted, it is done by taking a weighted sum
% (linear combination) of these orthonormal basis functions. Provided that
% the transmitter (Tx) and receiver (Rx) are both aware of the basis 
% functions being used, then a vector containing the weights used in the
% linear combination of ϕ1 and ϕ2 to form a given s(t) is sufficient to
% represent s(t). In this way, each s(t) can be represented by a vector
% with two components: the first component represents the amount of ϕ1
% needed to help construct the given s(t), and the second component
% indicates the amount of ϕ2 also needed to help construct the same s(t).
% 
% The x-component of each symbol is also referred to as the in-phase
% component - let's define a vector for the possible x-components of the
% possible symbols:
inPhase = -d/2:d:d/2; % vector containing possible x-components for all M symbols
% The y-component of each symbol is sometimes called the quadrature
% component - we can define another vector in the same way as above:
quadrature = -3*d/2:d:3*d/2; % vector containing possible y-components for all M symbols

% Now to form the set of M transmit symbols:
% The line below will help in getting each (x, y) pair in the
% constellation plot. Each (x, y) pair corresponds to a possible
% symbol:
[x, y] = meshgrid(inPhase, quadrature);
modulationAlphabet = x(:) + 1i*y(:); % the set of transmit symbols

% We can also quickly work out the average symbol energy (Es), now that
% we have our set of possible transmit symbols. Since each symbol is
% equally likely to be sent we could just sum up their respective energies
% and divide by M. However, in order to allow the symbols to possibly 
% have different probabilities of transmission we will calculate 
% the average energy as shown below - it is a weighted sum which involves 
% the probability of transmission of each symbol. Currently each of these 
% probabilities are the same, and equal to 1/M:
Es = sum(txProbabilities'.*abs(modulationAlphabet).^2);

EsNo = 10.^(-5:0.20:3); % a vector of values for (Es/No)...
EbNo = EsNo./log2(M); % a vector of the corresponding (Eb/No) values

No = Es./EsNo; % a vector of No values...

% Now we can generate a sequence of symbols to transmit. Each symbol
% should be equally likely to be picked from the set of symbols.
numSymbols = 2^17; % the amount of symbols in our data stream

s = zeros(numSymbols, 1); % a vector to hold our transmit symbols
SER = zeros(length(No), 1); % a vector to store SER values
BER = zeros(length(No), 1); % a vector to store BER values

if (bits2symbols == true)
    numBits = log2(M)*numSymbols; % number of bits we should generate
    txBits = randi([0 1], 1, numBits); % our bitstream...
    index = 0;
    for b = 1:log2(M):length(txBits)
        % Please note: I had tried creating a dictionary or hashmap where
        % the keys would be gray code sequences and the values would be
        % symbols from the modulation alphabet. This dictionary could then
        % be used in the following way: for every log2(M) bits we take from
        % the incoming bitstream, we get the corresponding symbol and load
        % it into the transmit sequence (s). I used a "container" from
        % MATLAB to do this, but it was quite slow, and the code was hard
        % to read. For practical reasons, I have decided to use a
        % switch-case statement instead. However, if the user has set
        % "bits2symbols" to be "true" then this section of the code does
        % not generalise well, and will need to be updated in order to
        % facilitate different values of M, as it is currently configured
        % to work with M = 8 only. Should the user which to change the value
        % of M, then "bits2Symbols" should be set to "false":
        index = index + 1;
        switch num2str(txBits(b:b+log2(M)-1)) % get next log2(M) bits
            % and then pick the relevant symbol from the modulation
            % alphabet and load it into the sequence s
            case "0  0  0"
                s(index) = modulationAlphabet(4); 
            case "0  0  1"
                s(index) = modulationAlphabet(8);
            case "0  1  0"
                s(index) = modulationAlphabet(3);
            case "0  1  1"
                s(index) = modulationAlphabet(7);
            case "1  1  0"
                s(index) = modulationAlphabet(2);
            case "1  1  1"
                s(index) = modulationAlphabet(6);
            case "1  0  0"
                s(index) = modulationAlphabet(1);
            case "1  0  1"
                s(index) = modulationAlphabet(5);
            otherwise
                disp("ERROR")
        end     
    end
else
    % Forming the transmit sequence of symbols (s) by randomly
    % sampling the set of transmit symbols (modulationAlphabet)
    % uniformly - this means each symbol is equally likely to be
    % picked. The third argument being "true" means there will be
    % replacement. This means our transmit sequence can have more than
    % one instance of a given symbol:
    s = randsample(modulationAlphabet, numSymbols, true); % s will have length "lengthData"
end

for v = 1:length(No)
    % Now to model the effect of the channel by adding White Gaussian Noise.
    % Since the in-phase and quadrature carriers are orthogonal, noise will
    % affect this independently. First, we create a noise vector of equal
    % length to the transmit sequence s:
    sigma = sqrt(No(v)/2);
    noisePowerWatts = sigma^2; % the noise power in Watts
    % The elements in the noise vector will have two components: one
    % component in the horizontal direction, and one in the vertical direction.
    % The horizontal components of the noise will cause pertubations to the
    % x-components of the transmit symbols. And the vertical components will
    % causes pertubations to the y-components of the transmit symbols:
    n = sigma*randn(numSymbols, 1) + sigma*1i*randn(numSymbols, 1);
    
    % Now we can form r, the received sequence:
    r = s + n; % by adding n, we are overlaying each transmit symbol with WGN
    
    % And now to try and implement the MAP decision rule at the receiver:
    % The vector C below precomputes part of the expression for the MAP rule.
    % Each element is computing according to:
    %
    % (1/2)(No)log(P(m(i))) - (1/2)||s(i)||^2
    %
    % Where P(m(i)) is the probability of the ith symbol m(i) being sent.
    % And where ||s(i)||^2 is the magnitude of symbol s(i) - this equals the
    % the energy of s(i):
    C = (1/2)*No(v).*log10(txProbabilities) - (1/2)*(abs(modulationAlphabet)').^2;
    
    mHats = zeros(numSymbols, 1); % a vector for the Rx to store its decisions
    results = zeros(M, 1); % a vector to store intermediate results
    
    for i = 1:length(r)
        % The long expression below is from the lecture notes. In the first
        % part of the expression, we multiply and sum the real and complex parts of the
        % received symbol r(i) and each of the possible transmit symbols m(i).
        % This has the effect of computing the dot product between r(i) and each
        % of the possible transmit vectors. If two vectors are very close to
        % one another (i.e. pointing largely in the same direction) then
        % their dot product will be positive. As two vectors begin to diverge
        % or move apart, their dot product will reduce. As a result, the first
        % part of the expression below works out the "closeness" between what
        % was received r(i) and what could have been sent m(i)... we can then
        % factor in the constants in the vector C before making our decision:
        results = real(r(i)).*real(modulationAlphabet) + imag(r(i)).*imag(modulationAlphabet) + C';
        % Now let us get the index of the maximum argument:
        [maxVal, index] = max(results); % we only really need the index
        mHats(i) = modulationAlphabet(index); % add our decision to the mHats vector
    end
    
    % Working out the symbol error rate (SER):
    % We can create a logical vector made up of 1s and 0s
    % by comparing the transmitted sequence s with the mHats
    % sequence. If at a certain index, the receiver correctly
    % classifies the received signal, then there will be a 1. If
    % a received symbol is misclassified, then there will be a 0.
    % To create the logical vector we can do: mHat == s
    % Also, since a "1" corresponds to a correct decision by the
    % receiver, if we add up all the "1s" in the logical vector, and
    % subctract the number from the length of the vector we will end up
    % with the total number of incorrect decisions - i.e., the number of
    % errors made by our receiver:
    numSymbolErrors = numSymbols - sum(mHats == s);
    SER(v) = numSymbolErrors/numSymbols; % the symbol error rate (SER)
    
    % Now let us quickly work out the bit error rate (BER). First we will
    % need to translate the received symbols into bits. This would probably
    % be better done with a hashmap/dictionary, as this data structure has
    % a very fast lookup time. However, for practical reasons and also
    % because I am short on time I will use a switch-case statement:
    if (bits2symbols == true)
        j = 0; % a count variable used in the for loop below:
        rxBits = zeros(1, numBits); % a vector to store the bits received
        for e = 1:length(mHats)
            switch mHats(e)
                % The mapping below from symbols to bits agrees with the gray
                % code scheme that I have mentioned in my report:
                case modulationAlphabet(1)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [1 0 0];
                    j = j + 1;
                case modulationAlphabet(2)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [1 1 0];
                    j = j + 1;
                case modulationAlphabet(3)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [0 1 0];
                    j = j + 1;
                case modulationAlphabet(4)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [0 0 0];
                    j = j + 1;
                case modulationAlphabet(5)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [1 0 1];
                    j = j + 1;
                case modulationAlphabet(6)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [1 1 1];
                    j = j + 1;
                case modulationAlphabet(7)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [0 1 1];
                    j = j + 1;
                case modulationAlphabet(8)
                    rxBits(1, 1+j*log2(M):j*log2(M)+log2(M)) = [0 0 1];
                    j = j + 1;
                otherwise
                    disp("ERROR")
            end
        end
        numBitErrors = numBits - sum(rxBits == txBits);
        BER(v) = numBitErrors/numBits;
    end
end

% Now we can compute the theoretical values for the SER:
theoreticalSER = (5/2)*qfunc(sqrt((1/3).*EsNo));

if (plotSERcurves == true)
    figure(1)
    semilogy(10*log10(EsNo), SER, '*', 'MarkerSize', 10, 'MarkerFaceColor', [0.5 1 0.1], 'LineStyle', '--', 'color', 'g')
    title("\fontsize{14}\fontname{Georgia}Symbol Error Rate (SER) Vs E_{s}/N_{0} (M = " + M + ", No. symbols transmitted = " + numSymbols + ")");
    xlabel('\fontname{Georgia}\bf E_{s}/N_{0} (dB)');
    ylabel('\fontname{Georgia}\bf SER');
    set(gca,'Fontname', 'Georgia');
    grid on

    hold on

    semilogy(10*log10(EsNo), theoreticalSER, '*', 'MarkerSize', 8, 'MarkerFaceColor', [0.4 1 1], 'LineStyle', '--', 'color', 'b')

    legend("\fontsize{14}\fontname{Georgia}\bfSimulated SER", "\fontsize{14}\fontname{Georgia}\bfTheoretical SER");
    hold off
end

if (plotBERcurve && bits2symbols)
    figure(2)
    semilogy(10*log10(EbNo), BER, '*', 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.2 0.7], 'LineStyle', '--', 'color', 'b')
    title("\fontsize{14}\fontname{Georgia}Bit Error Rate (BER) Vs E_{b}/N_{0} (M = " + M + ", No. bits transmitted = " + numBits + ")");
    xlabel('\fontname{Georgia}\bf E_{b}/N_{0} (dB)');
    ylabel('\fontname{Georgia}\bf BER');
    set(gca,'Fontname', 'Georgia');
    legend("\fontsize{14}\fontname{Georgia}\bfSimulated BER")
end