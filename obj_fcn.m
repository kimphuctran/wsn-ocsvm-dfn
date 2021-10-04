function [f,g]=obj_fcn(dmin,dmax,s)

    n=numel(dmin);
    f=0;g=0;
    for i=1:n
        f=f-2/n*exp(-dmin(i)/2/s^2)+2/n*exp(-dmax(i)/2/s^2);
        g=g-2/n*exp(-dmin(i)/2/s^2)*dmin(i)/s^3+2/n*exp(-dmax(i)/2/s^2)*dmax(i)/s^3;
    end
    
    try
        Js=evalin('base','Js');
    catch
        Js=[];
    end
    Js=[Js;s g f];
    assignin('base','Js',Js);

end