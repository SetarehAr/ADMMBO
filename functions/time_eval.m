
function  [c]=time_eval(Params)
    dropout_In=Params(1);
    dropout_Hi=Params(2);
    momentum=Params(3);
    weightDecay=Params(4);
    maxWeight=Params(5);
    learningRate=Params(6);
    decayRate=Params(7);
    nodeNum_L1=round(Params(8));
    nodeNum_L2=round(Params(9));
    nodeNum_L3=round(Params(10));
    active_choice=round(Params(11));

 
	paramsString = sprintf('\''-dIn\'' %f \''-dHi\'' %f \''-mo\'' %f \''-wD\'' %f \''-wM\'' %f \''-lr\'' %f \''-dr\'' %f \''-nNum1\'' %d \''-nNum2\'' %d \''-nNum3\'' %d  \''-actCh\'' %d \''-jobID\'' 1 ', dropout_In,dropout_Hi,momentum, weightDecay, maxWeight,learningRate,decayRate,nodeNum_L1,nodeNum_L2,nodeNum_L3,active_choice);
	
	commandString = 'python MINST_keras_remote_c.py';
	
	fullCommand = sprintf('%s %s', commandString, paramsString);
   % fprintf(fullCommand)
	
	[~,cmdout] = system(fullCommand);

	splitOut = strsplit(cmdout, '\n');
	
	output = splitOut{end-1};
    A=[];
	A =sscanf(output,'{\"c1\": %f}');
    if isempty(A)
        A =sscanf(output,'{\"c1\": %f}');
    end
    c =-A(1);
end
