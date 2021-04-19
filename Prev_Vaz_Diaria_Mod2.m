% ============================
% ============================
% ==== PREVISÃO DE VAZÕES ====
% ============================
% ============================
% Sem os Zeros
    
clear all
clc

% ==============================================================================================================================
% Carregando Dados e Parâmetros da RNA
% ==============================================================================================================================

load Dados

% Parâmetros da RNA

sh     = 20;       %---> Show
ep     = 1000;     %---> Epochs
go     = 1e-5;     %---> Goal

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
 
treal  = [A1966(1,4:end),A1967(1,4:end),A1972(1,4:end),A1973(1,4:end),...
     A1976(1,4:end),A1978(1,4:end),A1979(1,4:end),A1982(1,4:end),A1987(1,4:end)];
 
ind1 = find(p1>1);
ind2 = find(p2>1);
ind3 = find(t>1);
index = ind3;
p = [p1(:,ind3);p2(:,ind3);p3(:,ind3);p4(:,ind3);p5(:,ind3);p6(:,ind3)];
t = t(:,ind3);

% ==============================================================================================================================
% ANN Modelling
% ==============================================================================================================================

    %---> Data Scaling (Mean=0; Sdt=1)
    [pn,psp] = mapstd(p);
    [tn,pst] = mapstd(t);
    
    TEST = 166+104+199;   %---> Number of Days for Test (1979,1982,1987) Dias com vazão
    VAL  = 192;   % 1976

    iitst=[size(p,2)-TEST+1:size(p,2)];          %---> Test
    iival=[size(p,2)-TEST-VAL+1:size(p,2)-TEST]; %---> Validation
    iitr =[1:size(p,2)-TEST-VAL];                %---> Training

    val.P=pn(:,iival);val.T=tn(:,iival);         %---> Defining Data Set Intervals
    test.P=pn(:,iitst);test.T=tn(:,iitst);
    ptr=pn(:,iitr);ttr=tn(:,iitr);
    
    net = newff(minmax(ptr),[10 1],{'purelin','purelin'},'trainscg');                  %---> Creating the ANN

    net.trainParam.show   = sh;                     %---> Implementing ANN Parameters
    net.trainParam.epochs = ep;
    net.trainParam.goal   = go;
    net=init(net);                                  %---> Inicializing Weights and Biases
    net.performFcn = 'mse';
    
    net = train(net,ptr,ttr,[],[],val,test);         %---> ANN Training

    yp  = sim(net,pn);                           %---> Simulating
    an  = (yp);                             
    ai = max(0,mapstd('reverse',an,pst));

    a = zeros(1,size(treal,2));
    a(index)=ai;
    
%===============================================================================================================================
% Results
%===============================================================================================================================
    %td = size(p,2);
    td = size(treal,2);
    tf = treal;
    
    figure(2)
    %X = [1:1:size(ind3,2)];                                           
    X = [1:1:size(treal,2)];                                           
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

    
% Efficiency

    F  = sum((tf(:,iitst)-a(:,iitst)).^2);
    Fo = sum((tf(:,iitst)-mean(tf(:,iitst))).^2);
    R2 = 1-(F/Fo)

% RMSE

    RMSE = sqrt(sum((tf(:,iitst)-a(:,iitst)).^2)/TEST);
    MSE  = RMSE^2;
    ER   = a(:,iitst)-tf(:,iitst);

    MAE  = mean(abs(tf(:,iitst)-a(:,iitst)));
aa = a(:,iitst);
tt = tf(:,iitst);
    I = find(aa>0);
    ErroPerc = mean(abs(tt(I)-aa(I))./tt(I));
    ErroPerc2 = sqrt(mean(((tt(I)-aa(I))./tt(I)).^2));
    NASH = 1-(sum((tt(I)-aa(I)).^2))/(sum((tt(I)-mean(tt(I))).^2));
    
    %I = find(a(:,iitst)>0);
    
    B    = mean(aa)-mean(tt)
    MSE  = mean((aa-tt).^2)
    RMSE = sqrt(MSE)
    V    = MSE-B^2
    RB   = B/mean(tt)
    MAE  = mean(abs(aa-tt))
    RMAE = MAE/mean(tt)
    Eff  = 1-(MSE/V)
    NASH = 1-(sum((tt(I)-aa(I)).^2))/(sum((tt(I)-mean(tt(I))).^2))
  format bank;

%   [SUCCESS,MESSAGE]=xlswrite('RESULT',[tf' a'],'Data')

save Result2;
