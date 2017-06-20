
#include <lamtram/lamtram-train.h>
#include <dynet/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    bool mp = false; // check whether we need shared memory:
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--mp" && i+1 < argc) { mp = (atoi(argv[i+1]) > 1); break; }
    }
    dynet::initialize(argc, argv, mp);
    LamtramTrain train;
    return train.main(argc, argv);
}
