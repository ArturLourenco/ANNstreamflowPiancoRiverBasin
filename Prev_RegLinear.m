% ==============================
% ==== OCORRÊNCIA DE VAZÕES ====
% ==============================
% ==============================

    
clear all
clc

% ==============================================================================================================================
% Carregando Dados e Parâmetros da RNA
% ==============================================================================================================================

load Dados

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

t2  = [A1966(1,4:end),A1967(1,4:end),A1972(1,4:end),A1973(1,4:end),...
     A1976(1,4:end),A1978(1,4:end),A1979(1,4:end),A1982(1,4:end),A1987(1,4:end)];

t1 = ceil(min(t2,1));

t = t2;
p = [p1;p2;p3;p4;p5];

% ==============================================================================================================================
% ANN Modelling
% ==============================================================================================================================

    %---> Data Scaling (Mean=0; Sdt=1)
    [pn,psp] = mapstd(p);
    [tn,pst] = mapstd(t);
    
    TEST = 365*3;   %---> Number of Days for Test 
    VAL  = 366;   %--->
    TRAIN = size(p,2)-TEST-VAL;
    
    iitst=[size(p,2)-TEST+1:size(p,2)];          %---> Test
    iival=[size(p,2)-TEST-VAL+1:size(p,2)-TEST]; %---> Validation
    iitr =[1:size(p,2)-TEST-VAL];                %---> Training

    val.P=pn(:,iival);val.T=tn(:,iival);         %---> Defining Data Set Intervals
    test.P=pn(:,iitst);test.T=tn(:,iitst);
    ptr=pn(:,iitr);ttr=tn(:,iitr);
    
    B = regress(t([iitr iival])',[p(:,[iitr iival])' ones(TRAIN+VAL,1)]);
    ap = max([p' ones(TRAIN+VAL+TEST,1)]*B,0);    
    ap = ap';
    
    figure(2)
    X=linspace(1,size(p',1),size(p',1));
    plot(X,t,'r-',X,ap,'b-');
    
    figure(3)
    [m,b,r1] = postreg(ap,t);
    
    figure(4)
    X=linspace(1,TEST,TEST);
    plot(X,t(iitst),'r-',X,ap(iitst),'b-');
    
    figure(5)
    [m,b,r2] = postreg(ap(iitst),t(iitst));
    
    Bias = mean(ap-t)*1000

    RESULT=[t' ap']
    
save RESULT_MLR;

[SUCCESS,MESSAGE] = xlswrite('ResultRLM',RESULT,'Dados','A2:B5809')

% ANÁLISE PARA PREVISÃO DE VAZÃO

