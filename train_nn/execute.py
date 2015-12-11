import numpy
import subprocess, shlex
def subprocess_cmd(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout
def throw_sleep(sec):
    bashCommand = 'sleep '+str(sec)
    process = subprocess_cmd(bashCommand)

if __name__=='__main__':
    epsilons = [0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0]

    for epsilon in epsilons:
        throw_sleep(10.0)
        bashCommand = 'qsh python train_sup.py' \
                      + ' --cost_type=VAT'\
                      + ' --epsilon=' + str(epsilon)\

        process = subprocess_cmd(bashCommand)

    for epsilon in epsilons:
        throw_sleep(10.0)
        bashCommand = 'qsh python train_sup.py' \
                      + ' --cost_type=VAT_finite_diff'\
                      + ' --epsilon=' + str(epsilon)\

        process = subprocess_cmd(bashCommand)

    for epsilon in epsilons:
        throw_sleep(10.0)
        bashCommand = 'qsh python train_semisup.py' \
                      + ' --cost_type=VAT'\
                      + ' --epsilon=' + str(epsilon)\

        process = subprocess_cmd(bashCommand)

    for epsilon in epsilons:
        throw_sleep(10.0)
        bashCommand = 'qsh python train_semisup.py' \
                      + ' --cost_type=VAT_finite_diff'\
                      + ' --epsilon=' + str(epsilon)\

        process = subprocess_cmd(bashCommand)


    throw_sleep(10.0)
    bashCommand = 'qsh python train_sup.py' \
                  + ' --cost_type=MLE'\

    process = subprocess_cmd(bashCommand)
