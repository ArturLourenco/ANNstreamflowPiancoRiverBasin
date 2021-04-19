clear all
clc

% ============================
% ============================
% ==== PREVISÃO DE VAZÕES ====
% ============================
% ============================
% Com os Zeros

%======================================================================================================================================&

%== Dados da Estação Fluviométrica Piancó ==
%== Dados das Estações Pluviométrica disponivéis na Sub-bacia do Rio Piancó ==

% ==============================================================================================================================
% Carregando Dados e Parâmetros da RNA
% ==============================================================================================================================

load Dados

%======================================%
%== Aquisição e Tratamento dos Dados ==%
%======================================%

p1 = [A1966(1,1:end-3),A1967(1,1:end-3),A1972(1,1:end-3),A1973(1,1:end-3),...
     A1976(1,1:end-3),A1978(1,1:end-3),A1979(1,1:end-3),A1982(1,1:end-3),A1987(1,1:end-3)];

p2 = [A1966(1,2:end-2),A1967(1,2:end-2),A1972(1,2:end-2),A1973(1,2:end-2),...
     A1976(1,2:end-2),A1978(1,2:end-2),A1979(1,2:end-2),A1982(1,2:end-2),A1987(1,2:end-2)];

p3  = [A1966(1,3:end-1),A1967(1,3:end-1),A1972(1,3:end-1),A1973(1,3:end-1),...
     A1976(1,3:end-1),A1978(1,3:end-1),A1979(1,3:end-1),A1982(1,3:end-1),A1987(1,3:end-1)];

p4 = [A1966([3 6:7],2:end-2),A1967([3 6:7],2:end-2),A1972([3 6:7],2:end-2),A1973([3 6:7],2:end-2),...
     A1976([3 6:7],2:end-2),A1978([3 6:7],2:end-2),A1979([3 6:7],2:end-2),A1982([3 6:7],2:end-2),A1987([3 6:7],2:end-2)];

p5 = [A1966(3:7,3:end-1),A1967(3:7,3:end-1),A1972(3:7,3:end-1),A1973(3:7,3:end-1),...
     A1976(3:7,3:end-1),A1978(3:7,3:end-1),A1979(3:7,3:end-1),A1982(3:7,3:end-1),A1987(3:7,3:end-1)];

p6 = [A1966([3 6],4:end),A1967([3 6],4:end),A1972([3 6],4:end),A1973([3 6],4:end),...
     A1976([3 6],4:end),A1978([3 6],4:end),A1979([3 6],4:end),A1982([3 6],4:end),A1987([3 6],4:end)];

t  = [A1966(1,4:end),A1967(1,4:end),A1972(1,4:end),A1973(1,4:end),...
     A1976(1,4:end),A1978(1,4:end),A1979(1,4:end),A1982(1,4:end),A1987(1,4:end)];
 
p = [p1;p2;p3;p4;p5;p6];
t = t;


inputSeries = tonndata(p,true,false);%<-- Transforma os dados de entrada da forma double para cell.
targetSeries = tonndata(t,true,false);%<-- Transforma os dados alvo da forma double para cell.

%==================================================%
%== Criar, configurar e inicializar a Rede Neural==%
%==================================================%

net = feedforwardnet;%<-- Cria uma rede neural alimentada adiante. 
net.layers{1}.size = 4;%<-- Numero de neuronios da camada oculta.
net.layers{2}.size = 1;%<-- Numero de neuronios da camada de saida.
net = configure(net,inputSeries,targetSeries);%<-- Define as estradas e os alvos da RNA.

%== Pré/Pós processamento do conjunto de dados

net.inputs{1}.processFcns = {'mapstd','mapminmax'};%<-- Função para normalização dos dados de entrada. 
net.outputs{2}.processFcns = {'mapstd','mapminmax'};%<-- Função para normalização dos dados de saida.

%== Dividindo o conjundo de dados

net.divideFcn = 'dividerand';%<-- Divide o conjunto de dados de forma intercalada.
net.divideMode = 'sample';%<-- ???.
net.divideParam.trainRatio = 70/100;%<-- 70% treino.
net.divideParam.valRatio = 15/100;%<-- 15% para validação.
net.divideParam.testRatio = 15/100;%<-- 15% para teste.

%== Define as configurações da rede

net.trainFcn = 'trainlm';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net.performFcn = 'mse'; 

%== Implementa os parametros de treinamento

net.trainParam.epochs =	200;	    %Maximum number of epochs to train
net.trainParam.goal = 1e-5;	        % Performance goal
net.trainParam.max_fail = 5;	    %Maximum validation failures
net.trainParam.min_grad = 1e-10;	%Minimum performance gradient
net.trainParam.mu =	0.001;	        %Initial mu
net.trainParam.mu_dec =	0.1;	    %mu decrease factor
net.trainParam.mu_inc =	10;	        %mu increase factor
net.trainParam.mu_max =	1e10;	    %Maximum mu
net.trainParam.show = 10;	        %Epochs between displays (NaN for no displays)
net.trainParam.showCommandLine = 0;	%Generate command-line output
net.trainParam.showWindow = 1;	    %Show training GUI
net.trainParam.time = inf;	        %Maximum time to train in seconds

net=init(net);%<-- Inicializa os Weights e Biases. 

[net,tr] = train(net,inputSeries,targetSeries);%<-- Inicia o treinamento da rede.

%==================%
% Testa a Rede ====%
%==================%

outputs = net(inputSeries);%<-- As saidas será o resultado do treinamento da "net" a partir das entradas.
errors = gsubtract(targetSeries,outputs);%<--- Calcula os erros.
performance = perform(net,targetSeries,outputs);%<-- Calcula a performace.

%===========================================================%
% Recalcula a performace do treinamento, validação e teste =%
%===========================================================%

trainTargets = gmultiply(targetSeries,tr.trainMask);%<-- Separa os dados de Treinamento.
valTargets = gmultiply(targetSeries,tr.valMask);%<-- Separa os dados de Validação.
testTargets = gmultiply(targetSeries,tr.testMask);%<-- Separa os dados de Teste.
trainPerformance = perform(net,trainTargets,outputs);%<-- Calcula a performace de Treinamento.
valPerformance = perform(net,valTargets,outputs);%<-- Calcula a performace de Validação.
testPerformance = perform(net,testTargets,outputs);%<-- Calcula a performace Teste.

%===============================================================================================================================
% Resultados e Gráficos
%===============================================================================================================================

figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotresponse(targetSeries,outputs)
figure, ploterrcorr(errors)
figure, plotinerrcorr(inputSeries,errors)

trOut = outputs(tr.trainInd);%<-- Retira os indices para dados de saida do Treinamento.
vOut = outputs(tr.valInd);%<-- Retira os indices para dados de saida da Validação.
tsOut = outputs(tr.testInd);%<-- Retira os indices para dados de saida do Teste.
trTarg = targetSeries(tr.trainInd);%<-- Retira os indices para dados de entrada do Treinamento.
vTarg = targetSeries(tr.valInd);%<-- Retira os indices para dados de entrada do Treinamento.
tsTarg = targetSeries(tr.testInd);%<-- Retira os indices para dados de entrada do Treinamento.
wholeTarg = targetSeries;%<-- Conjunto completo de dados entrada. 
wholeOut = outputs;%<-- Conjunto completo de dados saida.

a = cell2mat(outputs);
t = cell2mat(targetSeries);

 figure(3)
    [m,b,r1] = postreg(a,t);

% plotregression(trTarg,trOut,'Train',vTarg,vOut,'Validation',...% Plota a regressão para todos os conjuntos de dados.
% tsTarg,tsOut,'Testing',wholeTarg,wholeOut,'Whole Dataset')     %

save Result3;
