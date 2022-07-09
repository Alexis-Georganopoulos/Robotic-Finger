function [data_mat] = litcount(filename)
% Extracts the pdc/electrode/z_dir data from the biochip ROS package

global data_mat;
data_mat = [];
counter = 1;
fid = fopen(filename);

tline = fgetl(fid);

while ischar(tline)
    if(contains(tline, 'pdc: ')) % or strfind
        tline = fgetl(fid);
        pdc_extract(tline,counter);
    end
    
    if(contains(tline, 'electrode: ')) % or strfind
        tline = fgetl(fid);
        electrode_extract(tline,counter);
    end
    
    if(contains(tline, 'z_dir: ')) % or strfind
        tline = fgetl(fid);
        tline2 = fgetl(fid);
        tline3 = fgetl(fid);
        z_dir_extract(tline,tline2,tline3,counter);
        counter = counter + 1;
    end
    tline = fgetl(fid);
end
fclose(fid);
end

function pdc_extract(line,counter)
    
    global data_mat;
    data_mat(counter,1) = str2num(line(1:end));

end

function electrode_extract(line, counter)

    global data_mat;
    match = [","  ,  "("  ,  ")"];
    temp = erase(line,match);
    temp = str2num(temp);
    data_mat(counter,2:20) = temp(1:end);

end

function z_dir_extract(line, line2, line3, counter)

    global data_mat;
    match = [  "["  ,  "]"  ];
    temp = erase(line,match);
    temp = str2num(temp);
    temp2 = erase(line2,match);
    temp2 = str2num(temp2);
    temp3 = erase(line3,match);
    temp3 = str2num(temp3);
    
    data_mat(counter,21:23) = [temp temp2 temp3];
    
end
