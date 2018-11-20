import math, sys, time
import pp
import centro_main


def generate_loc(A, id_level, delta_bp, size_filter_bp):
    res = A.single_bootstrap(id_level, delta_bp, size_filter_bp)
    return res


class parallel:
    def __init__(self, ncpus, analysis_obj):
        # Create jobserver
        self.job_server = pp.Server()
        self.job_server.set_ncpus(ncpus)
        self.jobs = []
        self.A = analysis_obj
        print("object created")

    def launch_computing(self, n_bootstrap, id_level, delta_bp, size_filter_bp):
        start_time = time.time()
        for j in range(0, n_bootstrap):
            # Submit a job which will calculate partial sum
            # part_sum - the function
            # (id_level, delta_bp, size_filter_bp) - tuple with arguments for part_sum
            # () - tuple with functions on which function part_sum depends
            # (CM) - tuple with module names which must be imported before part_sum execution

            self.jobs.append(
                self.job_server.submit(
                    generate_loc,
                    (self.A, id_level, delta_bp, size_filter_bp),
                    (),
                    ("centro_main",),
                )
            )
            print("sob submited")
        self.job_server.wait()
