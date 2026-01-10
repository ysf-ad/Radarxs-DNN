#include "radarxs.h"

#define Env Radarxs
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->initial_targets = unpack(kwargs, "initial_targets");
    env->max_trackers = unpack(kwargs, "max_trackers");
    if (env->initial_targets == 0) env->initial_targets = 30; // Default
    if (env->max_trackers == 0) env->max_trackers = 500; // Default
    
    // Allocate targets array
    env->targets = (Target*)calloc(env->max_trackers, sizeof(Target));
    if (!env->targets) {
        printf("Failed to allocate targets!\n");
        return -1;
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
