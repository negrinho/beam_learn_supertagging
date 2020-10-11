# NOTE: uncomment for having commands ran echoed to the terminal.
# set -x

ut_random_uuid() { python -c "import uuid; print(uuid.uuid4())"; }

ut_run_command_on_server(){ ssh "$2" -t "$1"; }
ut_run_command_on_server_on_folder(){ ssh "$2" -t "cd \"$3\" && $1"; }
ut_run_bash_on_server_on_folder(){ ssh "$1" -t "cd \"$2\" && bash"; }

UT_RSYNC_FLAGS="--archive --update --recursive --verbose"
ut_sync_file_to_server(){ rsync $UT_RSYNC_FLAGS "$1" "$2"; }
ut_sync_file_from_server(){ rsync $UT_RSYNC_FLAGS "$1" "$2"; }
ut_sync_folder_to_server(){ rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }
ut_sync_folder_from_server(){ rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }

# NOTE: excluding certain directories may be desirable if syncing them
# them frequently becomes a bottleneck or it is not desirable to have
# exactly the same folders replicated locally and in the server (e.g.,
# very large data). see here for info on how to accomplish this.
# https://www.thegeekstuff.com/2011/01/rsync-exclude-files-and-folders/

######## TODO:
# this part is to filled with information about your credentials and where your project will live.
# these functions were developed for the bridges supercomputing platform, but should be easily adaptable for other SLURM-based clusters.
USERNAME="USERNAME"
USER_HOST="${USERNAME}@bridges.psc.xsede.org"
LOCAL_FOLDERPATH=ABSOLUTE_PATH_TO_LOCAL_PROJECT_FOLDER
REMOTE_FOLDERPATH=ABSOLUTE_PATH_TO_REMOTE_PROJECT_FOLDER
########


### commands are run with respect to the specific project folder.
utproj_sync_file_to_server(){ ut_sync_file_to_server "$LOCAL_FOLDERPATH/$1" "$USER_HOST:$REMOTE_FOLDERPATH/$1"; }
utproj_sync_file_from_server(){ ut_sync_file_from_server "$USER_HOST:$REMOTE_FOLDERPATH/$1" "$LOCAL_FOLDERPATH/$1"; }
utproj_sync_folder_to_server(){ ut_sync_folder_to_server "$LOCAL_FOLDERPATH/$1" "$USER_HOST:$REMOTE_FOLDERPATH/$1"; }
utproj_sync_folder_from_server(){ ut_sync_folder_from_server "$USER_HOST:$REMOTE_FOLDERPATH/$1" "$LOCAL_FOLDERPATH/$1"; }

utproj_run_command_on_server(){ ut_run_command_on_server "$1" "$USER_HOST"; }
utproj_run_command_on_server_on_project_folder(){ ut_run_command_on_server_on_folder "$1" "$USER_HOST" "$REMOTE_FOLDERPATH"; }
utproj_run_bash_on_server(){ utproj_run_command_on_server "bash"; }

### commands for slurm.
utproj_show_queue(){ utproj_run_command_on_server "squeue"; }
utproj_show_my_jobs(){ utproj_run_command_on_server "squeue -u $USERNAME"; }
utproj_cancel_job(){ utproj_run_command_on_server "scancel -n \"$1\""; }
utproj_cancel_all_my_jobs(){ utproj_run_command_on_server "scancel -u $USERNAME"; }

# 1: command, 2: job name, 3: folder, 4: num cpus, 5: memory in mbs, 6: time in minutes
# limits: 4GB per cpu, 48 hours,
# NOTE: read https://www.psc.edu/bridges/user-guide/running-jobs for more details.
utproj_submit_cpu_job_with_resources(){
    script='#!/bin/bash'"
#SBATCH --nodes=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=$4
#SBATCH --mem=$5MB
#SBATCH --time=$6
#SBATCH --job-name=\"$2\"
$1" && utproj_run_command_on_server "cd \"$3\" && echo \"$script\" > _run.sh && chmod +x _run.sh && sbatch _run.sh && rm _run.sh";
}

# NOTE: basic resource usage for testing. change to suit your project.
# NOTE: it may be useful to define multiple commands to use different
# amounts of resources.
utproj_run_server_cpu_command(){ utproj_submit_cpu_job_with_resources "$1" "my_cpu_job" "$REMOTE_FOLDERPATH" 1 1024 60; }

# registering a ssh key on the server.
# NOTE: a more complex procedure may be necessary for the server being used.
ut_register_ssh_key_on_server(){ ssh-copy-id "$1"; }

# NOTE: watch may need to be installed first.
ut_run_command_every_num_seconds(){ watch -n "$2" "$1"; }
utproj_continuously_sync_project_from_server(){ ut_run_command_every_num_seconds "source utils.sh && utproj_sync_folder_from_server ." 300; }
utproj_sync_project_to_server(){ utproj_sync_folder_to_server "."; }
utproj_sync_project_from_server(){ utproj_sync_folder_from_server "."; }

utproj_delete_file(){ rm "$1" && utproj_run_command_on_server_on_project_folder "rm $1"; }
utproj_delete_folder(){ rm -r "$1" && utproj_run_command_on_server_on_project_folder "rm -r $1"; }

# NOTE: convenient shorter aliases for most common commands.
# user may find useful to define additional ones.
srv_cmd(){ utproj_run_command_on_server_on_project_folder "$1"; }
# srv_cpu_cmd(){ utproj_run_server_cpu_command_in_singularity_container "$1"; }
# srv_gpu_cmd(){ utproj_run_server_gpu_command_in_singularity_container "$1"; }
srv_sync(){ utproj_sync_project_to_server && utproj_sync_project_from_server; }

ut_create_conda_environment() { conda create --name "$1"; }
ut_create_conda_py27_environment() { conda create --name "$1" python=2.7; }
ut_create_conda_py36_environment() { conda create --name "$1" python=3.6; }
ut_show_conda_environments() { conda info --envs; }
ut_show_installed_conda_packages() { conda list; }
ut_delete_conda_environment() { conda env remove --name "$1"; }
ut_activate_conda_environment() { conda activate "$1"; }
ut_deactivate_conda_environment() { conda deactivate; }

# 1: command, 2: job name, 3: folder, 4: num cpus, 5: memory in mbs, 6: time in minutes 7: output_folder
utproj_run_main_with_config(){
    utproj_submit_cpu_job_with_resources \
        "module load anaconda2/5.1.0 && source activate beam_learn && python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --train --config_filepath configs/cfg$1.json" \
        "cfg$1" "$REMOTE_FOLDERPATH" 2 8000 2400;
}

utproj_run_main_locally_with_config(){
    python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --train --config_filepath configs/cfg$1.json;
}

utproj_run_main_with_configs_from_to(){
    for i in $(eval echo {$1..$2})
    do
        echo "cfg$i"
        utproj_run_main_with_config $i;
    done
}

utproj_run_main_vanilla_beam_accuracy_of_config(){
    python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --compute_vanilla_beam_accuracy --config_filepath configs/cfg$1.json;
}

utproj_run_main_beam_accuracy_of_config(){
    python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --compute_beam_accuracy --config_filepath configs/cfg$1.json;
}

# --------------------------- RESULT GENERATION ---------------------------
# 1: command, 2: job name, 3: folder, 4: num cpus, 5: memory in mbs, 6: time in minutes 7: output_folder
utproj_run_main_with_config_with_repeat(){
    utproj_submit_cpu_job_with_resources \
        "module load anaconda2/5.1.0 && source activate beam_learn && python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --train --config_filepath configs/cfg_r$2_$1.json" \
        "cfg_r$2_$1" "$REMOTE_FOLDERPATH" 2 8000 2400;
}

utproj_run_main_locally_with_config_with_repeat(){
    python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --train --config_filepath configs/cfg_r$2_$1.json;
}

# NOTE: compared with the other one, this one uses the fact that the
utproj_run_main_with_configs_with_repeat_from_to(){
    for i in $(eval echo {$1..$2})
    do
        echo "cfg_r$3_$i"
        utproj_run_main_with_config_with_repeat $i $3;
    done
}

utproj_run_main_with_configs_with_repeat_from_to_double(){
    for j in $(eval echo {$3..$4})
    do
        for i in $(eval echo {$1..$2})
        do
            echo "cfg_r${j}_${i}"
            utproj_run_main_with_config_with_repeat $i $j;
        done
    done
}

utproj_run_main_vanilla_beam_accuracy_of_config_double(){
    for j in $(eval echo {$3..$4})
    do
        for i in $(eval echo {$1..$2})
        do
            echo "cfg_r${j}_${i}.json"
            python -u main.py --dynet-mem 4000 --dynet-autobatch 1 --compute_vanilla_beam_accuracy --config_filepath configs/cfg_r${j}_${i}.json;
        done
    done
}

# this function uses three repeats is done for three repeats (i.e., the number of repeats used in the paper)
utproj_table1_results(){
    utproj_run_main_vanilla_beam_accuracy_of_config_double 1000 1005 0 2;
}

# # three repeats (config ranges below)
# # 1000 1005
# # 2000 2039
# # 3000 3055
# # 4000 4047
# # 5000 5047
# # 6000 6039
