% ============================
% ============================
% ==== PREVISÃO DE VAZÕES ====
% ============================
% ============================

    
clear all
clc

% ==============================================================================================================================
% Carregando Dados e Parâmetros da RNA
% ==============================================================================================================================

load Dados

% Transformando Dados para 1's e 0's

A1966(1,:) = ceil(min(A1966(1,:),1));
A1967(1,:) = ceil(min(A1967(1,:),1));
A1972(1,:) = ceil(min(A1972(1,:),1));
A1973(1,:) = ceil(min(A1973(1,:),1));
A1976(1,:) = ceil(min(A1976(1,:),1));
A1978(1,:) = ceil(min(A1978(1,:),1));
A1979(1,:) = ceil(min(A1979(1,:),1));
A1982(1,:) = ceil(min(A1982(1,:),1));
A1987(1,:) = ceil(min(A1987(1,:),1));

% Parâmetros da RNA

sh     = 10;       %---> Show
ep     = 2000;     %---> Epochs
go     = 1e-3;     %---> Goal

% ==============================================================================================================================
% ANN Data (Input "p" and Target "t")
% ==============================================================================================================================

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


% ==============================================================================================================================
% ANN Modelling
% ==============================================================================================================================

    %---> Data Scaling (Mean=0; Sdt=1)
    [pn,psp] = mapstd(p);
    [tn,pst] = mapstd(t);
    
    TEST = 365*3;   %---> Number of Days for Test (1979,1982,1987)
    VAL  = 366;     % 1976
    TRAIN = size(p,2)-TEST-VAL;
    
    iitst=[size(p,2)-TEST+1:size(p,2)];          %---> Test
    iival=[size(p,2)-TEST-VAL+1:size(p,2)-TEST]; %---> Validation
    iitr =[1:size(p,2)-TEST-VAL];                %---> Training

    val.P=pn(:,iival);val.T=tn(:,iival);         %---> Defining Data Set Intervals
    test.P=pn(:,iitst);test.T=tn(:,iitst);
    ptr=pn(:,iitr);ttr=tn(:,iitr);
    
    net = newff(minmax(ptr),[5 1],{'tansig','purelin'},'trainlm');                  %---> Creating the ANN

    net.trainParam.show   = sh;                     %---> Implementing ANN Parameters
    net.trainParam.epochs = ep;
    net.trainParam.goal   = go;
    net=init(net);                                  %---> Inicializing Weights and Biases
    net.performFcn = 'mse';
    
    net = train(net,ptr,ttr,[],[],val,test);         %---> ANN Training

    yp  = sim(net,pn);                           %---> Simulating
    an  = (yp);                             
    a = round(min(max(0,mapstd('reverse',an,pst)),1));

%===============================================================================================================================
% Results
%===============================================================================================================================
    td = size(p,2);
    tf = t;
    
    figure(2)
    X = [1:1:td];                                           
    plot(X,tf,'r-',X,a,'b-');                                      %---> Relationship for the Whole Data Set
    
    figure(3)
    [m,b,r1] = postreg(a,tf);                                      %---> Correlation for the Whole Data Set
    
    figure(4)
    X = linspace(1,TEST,TEST);                                         
    plot(X,tf(:,iitst),'r-',X,a(:,iitst),'b-');                    %---> Relationship for the Test Data Set
    
    figure(5)
    [m,b,r2] = postreg(a(:,iitst),tf(:,iitst));                    %---> Correlation for the Test Data Set    
    

%===============================================================================================================================
% Statistical Indices
%===============================================================================================================================

% Porcentagem de Acertos

AcertoTrain = 0;
AcertoVal = 0;
AcertoTest  = 0;

for j =1:TRAIN
    if a(j)==t(j)
        AcertoTrain = AcertoTrain+1;
    end
end
for j =TRAIN+1:TRAIN+VAL
    if a(j)==t(j)
        AcertoVal = AcertoVal+1;
    end
end
for j =TRAIN+VAL+1:TRAIN+VAL+TEST
    if a(j)==t(j)
        AcertoTest = AcertoTest+1;
    end
end

AcTrain = 100*AcertoTrain/TRAIN;
AcVal = 100*AcertoVal/VAL;

AcCal = 100*(AcertoTrain+AcertoVal)/(TRAIN+VAL)
AcTest = 100*AcertoTest/TEST

dados= a';

% [SUCCESS,MESSAGE] = xlswrite('ResultFinalSeparado',dados,'Dados','A2:A3261')
% [SUCCESS,MESSAGE] = xlswrite('ResultFinalSeparado',[AcCal;AcTest]/100,'Acertos','B2:B3')
% 
% save Resultmodocorr;
    