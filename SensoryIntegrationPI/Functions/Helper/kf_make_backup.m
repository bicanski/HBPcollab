here = pwd;

notes = inputdlg('Enter notes:','Version Notes',20);

goto kalman
cd ../

if (exist('Kalman_Filter_Backups/backup_tmp')==7)
    system('rm -r ./Kalman_Filter_Backups/backup_tmp')
end

date = seshdate; date=date(1:8);

D = dir(['Kalman_Filter_Backups/Backup_',date,'*']);
foldname = ['Backup_',date,'_',num2str(length(D)+1)];

system('robocopy .\Kalman_Filter .\Kalman_Filter_Backups\backup_tmp /E /xf *.fig *.mat *.zip .*pdf');

fid = fopen('Kalman_Filter_Backups\backup_tmp\version_notes.txt','wt');
for l=1:size(notes{1},1)
    fprintf(fid, [notes{1}(l,:),'\n']);
end
fclose(fid);

clc; disp('Copying succesful')
disp('Zipping folder...')

zip(['Kalman_Filter_Backups/',foldname,'.zip'],'Kalman_Filter_Backups/backup_tmp/');

disp('Deleting temporary file (this may take a few minutes)...')

system('rm -r ./Kalman_Filter_Backups/backup_tmp')

disp('Done')

cd(here)