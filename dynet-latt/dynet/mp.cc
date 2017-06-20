#if !_WINDOWS
#include "mp.h"
using namespace std;
using namespace boost::interprocess;

namespace dynet {
  namespace mp {
    // TODO: Pass these around instead of having them be global
    std::string queue_name = "dynet_mp_work_queue";
    std::string shared_memory_name = "dynet_mp_shared_memory";
    timespec start_time;
    timespec start_time_last;
    bool stop_requested = false;
    SharedObject* shared_object = nullptr;

    void GetUTCTime(timespec & ts){
      #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
	  clock_serv_t cclock;
	  mach_timespec_t mts;
	  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	  clock_get_time(cclock, &mts);
	  mach_port_deallocate(mach_task_self(), cclock);
	  ts.tv_sec = mts.tv_sec;
	  ts.tv_nsec = mts.tv_nsec;
      #else
	  clock_gettime(CLOCK_REALTIME, &ts);
      #endif
    }
    double ElapsedTime() {
      timespec end_time;
      GetUTCTime(end_time);
      return (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
    }
    double ElapsedTimeDelta() {
      timespec end_time;
      GetUTCTime(end_time);
      double ret = (end_time.tv_sec - start_time_last.tv_sec) + (end_time.tv_nsec - start_time_last.tv_nsec) / 1000000000.0;
      GetUTCTime(start_time_last);
      return ret;
    }


    std::string generate_queue_name() {
      std::ostringstream ss;
      ss << "dynet_mp_work_queue";
      ss << rand();
      return ss.str();
    }

    std::string generate_shared_memory_name() {
      std::ostringstream ss;
      ss << "dynet_mp_shared_memory";
      ss << rand();
      return ss.str();
    }

    dynet::real sum_values(const std::vector<dynet::real>& values) {
      return accumulate(values.begin(), values.end(), 0.0);
    }

    dynet::real mean(const std::vector<dynet::real>& values) {
      return sum_values(values) / values.size();
    }

    std::string elapsed_time_string(const timespec& start, const timespec& end) {
      std::ostringstream ss;
      time_t secs = end.tv_sec - start.tv_sec;
      long nsec = end.tv_nsec - start.tv_nsec;
      ss << secs << " seconds and " << nsec << "nseconds";
      return ss.str();
    }

    unsigned spawn_children(std::vector<Workload>& workloads) {
      const unsigned num_children = workloads.size();
      assert (workloads.size() == num_children);
      pid_t pid;
      unsigned cid;
      for (cid = 0; cid < num_children; ++cid) {
        pid = fork();
        if (pid == -1) {
          std::cerr << "Fork failed. Exiting ..." << std::endl;
          return 1;
        }
        else if (pid == 0) {
          // children shouldn't continue looping
          break;
        }
        workloads[cid].pid = pid;
      }
      return cid;
    }

    std::vector<Workload> create_workloads(unsigned num_children) {
      int err;
      std::vector<Workload> workloads(num_children);
      for (unsigned cid = 0; cid < num_children; cid++) { 
        err = pipe(workloads[cid].p2c);
        assert (err == 0);
        err = pipe(workloads[cid].c2p);
        assert (err == 0);
      }
      return workloads;
    }

  }
}
#endif