#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

/////////////////////////////////////////////////////////////
// Const Dump Start

const Color color_bgteal = (Color){6, 24, 24, 255}; // Darkteal
const Color color_blue = (Color){0, 0, 255, 255}; // Blue
const Color color_sky = (Color){0, 128, 255, 255}; // Sky
const Color color_gray = (Color){128, 128, 128, 255}; // Gray
const Color color_red = (Color){255, 0, 0, 255}; // Red
const Color color_white = (Color){255, 255, 255, 255}; // White
const Color color_salmon = (Color){255, 85, 85, 255}; // Salmon
const Color color_silver = (Color){170, 170, 170, 255}; // Silver
const Color color_cyan = (Color){0, 255, 255, 255}; // Cyan
const Color color_yellow = (Color){255, 255, 0, 255}; // Yellow

const bool WRITE_LOGS_TO_FILES = false;
const unsigned char SEARCH = 0;
const unsigned char TRACK1 = 1;
const unsigned char TRACK2 = 2;
const unsigned char TRACK3 = 3;
const unsigned char TRACK4 = 4;
const unsigned char TRACK5 = 5;
const unsigned char NOOP = 6;

const int MAX_AZ_SLICES = 30;
const int MAX_EL_SLICES = 10;
const float AZ_DEGREES_PER_SLICE = 90.0f / MAX_AZ_SLICES;
const float EL_DEGREES_PER_SLICE = 30.0f / MAX_EL_SLICES;

const int MAX_SEARCHERS = 1;
const int FEATURES_PER_TRACKER = 4;  // t_desired, t_deadline, t_dwell_estimate, priority

const int PLACEHOLDER_FOR_SENSOR_ID = 1;

const float S_BAND_MAX_RANGE = 184000000.0f;  // 184 km in millimeters
const float X_BAND_MAX_RANGE = 100000000.0f;  // 100 km in millimeters
const float S_BAND_MIN_RANGE = 10000000.0f;   // 10 km in millimeters
const float X_BAND_MIN_RANGE = 5000000.0f;    // 5 km in millimeters

const float MAX_TARGET_XY_RANGE = 184000000.0f;  // 184 km in millimeters
const float MAX_TARGET_Z_RANGE = 20000000.0f;    // 20 km in millimeters
const float MAX_TARGET_XY_VELOCITY = 1000.0f;    // 1000 m/s
const float MIN_TARGET_SINGER_SIGMA = 0.0f;
const float MAX_TARGET_SINGER_SIGMA = 35.0f;
const float MIN_TARGET_SINGER_THETA = 1.0f;   // TODO: should this be 1000 milliseconds?
const float MAX_TARGET_SINGER_THETA = 50.0f;  // TODO: should this be 50000 milliseconds?
// Table III "Singer Manoeuvre Parameters for Three Target Types"
// From A. Charlish, K. Woodbridge, and H. Griffiths,
// ‘Phased array radar resource management using continuous double auction’,
// IEEE Transactions on Aerospace and Electronic Systems,
// vol. 51, no. 3, pp. 2212–2224, 2015.
// DOI. No. 10.1109/TAES.2015.130558.
// Type 1: BigSigma in [20,35], BigTheta in [10,20]
// Type 2: BigSigma in [0,5], BigTheta in [1,4]
// Type 3: BigSigma in [5,20], BigTheta in [30,50]
// For now I'm just doing (min, max) over all types.

const int PRIORITY_LEVELS = 3;

const int MIN_DWELL_TIME = 10;     // 10 milliseconds
const int MAX_DWELL_TIME = 200;    // 100 milliseconds
const int MAX_DEADLINE = 30000;     // 30000 milliseconds
const int MIN_REVISIT_TIME = 100;  // 100 milliseconds (page 192 of VKB)
const int SEARCH_DWELL_TIME = 10;  // 10 milliseconds (from Sunilas slides)

// For reset
const int ZERO_COST_SEARCH_TIME = MAX_AZ_SLICES * MAX_EL_SLICES * SEARCH_DWELL_TIME;
const int NO_TARGET = -1;

const unsigned int S_BAND_SENSOR = 0;
const unsigned int X_BAND_SENSOR = 1;

const float REFERENCE_DWELL_TIME = 10.0f;  //  10 milliseconds
const float REFERENCE_RANGE = 184000000.0f;
const float REFERENCE_CROSS_SECTION = 1.0f;
const float REFERENCE_SNR = 40.0f;

const float TRACK_UPDATE_REWARD = 0.1f;
const float SEARCH_REWARD = 0.1f;
const float TRACK_LOSS_PENALTY = 1.0f; // This is a coefficient by priority
const float TRACK_DELAY_PENALTY = 1.0f/1000.0f;  // Penalty per second to ms
const float SEARCH_PENALTY = 0.1f/1000.0f; // Penalty per second to ms

const float VERTICAL_MOTION_FACTOR = 0.1f;

const int WINDOW_X_PX = 480;
const int WINDOW_Y_PX = 270;

// Const Dump End
/////////////////////////////////////////////////////////////

typedef struct {
  float x;
  float x_velocity;
  float x_acceleration;
  float y;
  float y_velocity;
  float y_acceleration;
  float z;
  float z_velocity;
  float z_acceleration;
  float singer_sigma;  // maneuver standard deviation
  float singer_theta;  // maneuver time constant
  float priority;
  bool is_active;
  bool is_tracked;
} Target;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;


// Required struct named same as env 
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int tick;
    int s_band_t_until_free;
    int x_band_t_until_free;
    Target *targets;  // should I allow more targets than trackers?
    int initial_targets;
    int max_trackers;
} Radarxs;

void initialize_target(Radarxs *env, int target_index) {
  env->targets[target_index].x = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_RANGE);
  env->targets[target_index].y = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_RANGE);
  env->targets[target_index].z = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_Z_RANGE);

  while (
    sqrt(
    env->targets[target_index].x * env->targets[target_index].x +
    env->targets[target_index].y * env->targets[target_index].y +
    env->targets[target_index].z * env->targets[target_index].z) >
    S_BAND_MAX_RANGE
    ) {
    env->targets[target_index].x = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_RANGE);
    env->targets[target_index].y = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_RANGE);
    env->targets[target_index].z = (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_Z_RANGE);
  }

  
  env->targets[target_index].x_velocity =
      (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_VELOCITY);
  env->targets[target_index].y_velocity =
      (float)rand() / (float)((float)RAND_MAX / MAX_TARGET_XY_VELOCITY);
  env->targets[target_index].z_velocity = 0;  // targets spawn doing level flight
  env->targets[target_index].x_acceleration = 0;
  env->targets[target_index].y_acceleration = 0;
  env->targets[target_index].z_acceleration = 0;
  env->targets[target_index].singer_sigma =
      (float)rand() /
          (float)((float)RAND_MAX / (MAX_TARGET_SINGER_SIGMA - MIN_TARGET_SINGER_SIGMA)) +
      MIN_TARGET_SINGER_SIGMA;
  env->targets[target_index].singer_theta =
      (float)rand() /
          (float)((float)RAND_MAX / (MAX_TARGET_SINGER_THETA - MIN_TARGET_SINGER_THETA)) +
      MIN_TARGET_SINGER_THETA;
  env->targets[target_index].priority = rand() % PRIORITY_LEVELS;
  env->targets[target_index].is_active = false;
  env->targets[target_index].is_tracked = false;
}

float normal(float mean, float stddev) {
  float u1 = (float)rand() / (float)RAND_MAX;
  float u2 = (float)rand() / (float)RAND_MAX;
  // Apply Box-Muller transform
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
  return mean + stddev * z0;
}

void update_tracker(Radarxs *env, int tracker_id) {
  float target_range = sqrt(env->targets[tracker_id].x * env->targets[tracker_id].x +
                            env->targets[tracker_id].y * env->targets[tracker_id].y +
                            env->targets[tracker_id].z * env->targets[tracker_id].z);
  // TODO: replace 42 with a per-target rcs
  float target_cross_section = 4.2f;

  // @Sunila todo fix sigma_theta as well
  float sigma_theta = 1;  // I'm not sure how this translates to 3d, I also
                          // think this should be sensor dependent
  // This should also be sensor dependent
  float u = 0.3;  // Magic number from the slides.

  // Page 191 of VKB
  // There is a useful approximation for T [6]
  // "... T \cong 4.6 s and E[n] \cong 1.5 s"
  // "for small V_0T... T \cong 0.9 s and E[n] \cong 1.3 s"

  float new_t_desired =
      1000.0f *
      (0.4 *
       pow((target_range / 1000.0f * sigma_theta * sqrt(env->targets[tracker_id].singer_theta) /
            env->targets[tracker_id].singer_sigma),
           0.4) *
       pow(u, 2.4) / (1 + 0.5 * pow(u, 2)));  // 1000.0f is to convert to milliseconds
  if (new_t_desired < MIN_REVISIT_TIME) {
    new_t_desired = MIN_REVISIT_TIME;
  }
  if (new_t_desired > MAX_DEADLINE) {
    new_t_desired = MAX_DEADLINE;
  }
  float new_t_deadline = new_t_desired * (2 + env->targets[tracker_id].priority);
  if (new_t_deadline > MAX_DEADLINE) {
    new_t_deadline = MAX_DEADLINE;
  }
  // print the new_t_desired and new_t_deadline
  float new_t_dwell_estimate = (REFERENCE_DWELL_TIME * pow((target_range / REFERENCE_RANGE), 4) *
                                (REFERENCE_CROSS_SECTION / target_cross_section) * REFERENCE_SNR);
  if (new_t_dwell_estimate < MIN_DWELL_TIME) {
    new_t_dwell_estimate = MIN_DWELL_TIME;
  }
  if (new_t_dwell_estimate > MAX_DWELL_TIME) {
    new_t_dwell_estimate = MAX_DWELL_TIME;
  }
  if (WRITE_LOGS_TO_FILES) {
    FILE *file = fopen("./logs/new_t_desired.csv", "a");
    fprintf(file, "%d,%d,%f\n", env->tick, tracker_id, new_t_desired);
    fclose(file);
  }

  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER] =
      new_t_desired;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER + 1] =
      new_t_deadline;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER + 2] =
      new_t_dwell_estimate;
  // Priority: Use target priority (0-2) or -1 if not tracked
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER + 3] =
      (float)env->targets[tracker_id].priority;
  
}

void search_sector(Radarxs *env, int sector) {
  float az_min = (sector % MAX_AZ_SLICES) * (AZ_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float az_max = ((sector + 1) % MAX_AZ_SLICES) * (AZ_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float el_min = (sector / MAX_AZ_SLICES) * (EL_DEGREES_PER_SLICE)*M_PI / 180.0f;
  float el_max = ((sector + MAX_AZ_SLICES) / MAX_AZ_SLICES) * (EL_DEGREES_PER_SLICE)*M_PI / 180.0f;

  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active && !env->targets[i].is_tracked) {
      float az = atan2(env->targets[i].y, env->targets[i].x);
      float el = asin(env->targets[i].z / sqrt(env->targets[i].x * env->targets[i].x +
                                               env->targets[i].y * env->targets[i].y +
                                               env->targets[i].z * env->targets[i].z));
      if (az >= az_min && az < az_max && el >= el_min && el < el_max) {
        float target_range =
            sqrt(env->targets[i].x * env->targets[i].x + env->targets[i].y * env->targets[i].y +
                 env->targets[i].z * env->targets[i].z);
        float probability_of_detection = 0.0f;
        if (env->s_band_t_until_free == 0) {
          // Probability of detection is approximated as
          // 1-exp^(-BigConstant/range^4) for s-band, 10ms dwell on 1m
          // square target at 184km has 0.75 probability of detection
          // 1-exp(-(10e32/184_000_000^4)) = 0.53,
          if (target_range >= S_BAND_MIN_RANGE && target_range <= S_BAND_MAX_RANGE) {
            probability_of_detection =
                1 - exp((((-10e32 / target_range) / target_range) / target_range) / target_range);
          }
        } else {
          // 1-exp(-(10e31/100_000_000^4)) = 0.63
          if (target_range >= X_BAND_MIN_RANGE && target_range <= X_BAND_MAX_RANGE) {
            probability_of_detection =
                1 - exp((((-10e31 / target_range) / target_range) / target_range) / target_range);
          }
        }
        if ((float)rand() / (float)(RAND_MAX) < probability_of_detection) {
          env->targets[i].is_tracked = true;
          update_tracker(env, i);
        }
      }
    }
  }
  env->rewards[0] += SEARCH_REWARD;
  if (env->observations[sector] < 0) {
    env->rewards[0] += SEARCH_PENALTY * env->observations[sector];
  }
  env->observations[sector] = ZERO_COST_SEARCH_TIME;
}

void add_log(Radarxs* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Radarxs *env) {
  // Set all sectors to zero cost search time
  for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
    env->observations[i] = ZERO_COST_SEARCH_TIME;
  }

  // Set all trackers to NO_TARGET
  for (int i = 0; i < env->max_trackers * FEATURES_PER_TRACKER; i++) {
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i] = NO_TARGET;
  }

  // Set the sensor type to S_BAND_SENSOR
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
      S_BAND_SENSOR;

  env->tick = 0;
  env->s_band_t_until_free = 0;
  env->x_band_t_until_free = 0;

  for (int i = 0; i < env->max_trackers; i++) {
    initialize_target(env, i);
  }
  for (int i = 0; i < env->initial_targets; i++) {
    env->targets[i].is_active = true;
    env->targets[i].is_tracked = true;  // Mark as tracked
    update_tracker(env, i);  // Populate initial observations
  }
}


// Required function
void c_step(Radarxs* env) {
  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;

  if (action == SEARCH) {
    if (env->s_band_t_until_free == 0) {
      // Find the least recently used cluster of 4 sectors for S-band
      int least_recently_used_cluster = 0;
      for (int j = 0; j < MAX_EL_SLICES - 1; j++) {
        for (int i = 0; i < MAX_AZ_SLICES - 1; i++) {
          int current_cluster_value =
              env->observations[i + j * MAX_AZ_SLICES] +
              env->observations[i + j * MAX_AZ_SLICES + 1] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES + 1];

          int least_cluster_value =
              env->observations[least_recently_used_cluster % MAX_AZ_SLICES +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES + 1] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES +
                                1];

          if (current_cluster_value < least_cluster_value) {
            least_recently_used_cluster = i + j * MAX_AZ_SLICES;
          }
        }
      }

      // S-band searches four sectors, right and below the least recently used sector
      search_sector(env, least_recently_used_cluster);
      search_sector(env, (least_recently_used_cluster + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES));
      search_sector(
          env, (least_recently_used_cluster + MAX_AZ_SLICES) % (MAX_AZ_SLICES * MAX_EL_SLICES));
      search_sector(
          env, (least_recently_used_cluster + MAX_AZ_SLICES + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES));
      env->s_band_t_until_free = SEARCH_DWELL_TIME;
    } else {
      int least_recently_used_sector = 0;
      for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
        if (env->observations[i] < env->observations[least_recently_used_sector]) {
          least_recently_used_sector = i;
        }
      }
      // X-band searches the least recently used sector
      search_sector(env, least_recently_used_sector);

      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  }
  // else if (action == TRACK1 || action == TRACK2 || action == TRACK3 || action
  // == TRACK4 || action == TRACK5)
  else if (action <= env->max_trackers) {
    action -= 1;  // easier than -1 all over the place for indexing.

    if (!env->targets[action].is_tracked) {
      env->rewards[0] = -1.0f;
    } else {
      env->rewards[0] = TRACK_UPDATE_REWARD;
      // Penalize the delay in updating the tracker
      if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER] < 0) {
        env->rewards[0] -= (float)(env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                                                     action * FEATURES_PER_TRACKER] *
                                   TRACK_DELAY_PENALTY / (1 + env->targets[action].priority));
      }

      float target_range = sqrt(env->targets[action].x * env->targets[action].x +
                                env->targets[action].y * env->targets[action].y +
                                env->targets[action].z * env->targets[action].z);
      if (env->s_band_t_until_free == 0 && target_range < S_BAND_MAX_RANGE &&
          target_range > S_BAND_MIN_RANGE) {
        update_tracker(env, action);
      } else if (env->x_band_t_until_free == 0 && target_range < X_BAND_MAX_RANGE &&
                 target_range > X_BAND_MIN_RANGE) {
        update_tracker(env, action);
      } 
      if (env->s_band_t_until_free == 0) {
        env->s_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2];
      } else {
        // TODO: actually calculate t_dwell for x-band, for now, just
        // make it faster than s-band
        env->x_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2] /
            2;
      }
    }
  } else {
    // TODO: Throw error?
  }

  // Move simulation forward
  int delta_t = env->s_band_t_until_free;
  // env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER + 1]
  // =
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
      S_BAND_SENSOR;
  if (env->x_band_t_until_free < delta_t) {
    delta_t = env->x_band_t_until_free;
    // env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER +
    // 1]=
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
        X_BAND_SENSOR;

  }

  if (delta_t > 0) {
    env->tick += delta_t;
    env->s_band_t_until_free -= delta_t;
    env->x_band_t_until_free -= delta_t;
    for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
      env->observations[i] -= delta_t;
    }
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].is_tracked) {
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER] -= delta_t;
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 1] -= delta_t;
        // if the tracker has expired, lose the track and apply the penalty
        // drop the track if t_deadline < 0
        if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 1] < 0) {
          // log the loss and all target stats in a csv file
          // FILE *file = fopen("./target_stats.csv", "a");
          // fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f\n", env->targets[i].x, env->targets[i].y,
          //        env->targets[i].z, env->targets[i].x_velocity, env->targets[i].y_velocity,
          //        env->targets[i].z_velocity, env->targets[i].singer_sigma, env->targets[i].singer_theta,
          //        env->targets[i].priority);
          // fclose(file);
          // printf("Tracker %d lost\n", i);
          // printf("Target range: %f\n", sqrt(env->targets[i].x * env->targets[i].x +
          //                                  env->targets[i].y * env->targets[i].y +
          //                                  env->targets[i].z * env->targets[i].z));
          // printf("Target velocity: %f\n", sqrt(env->targets[i].x_velocity * env->targets[i].x_velocity +
          //                                    env->targets[i].y_velocity * env->targets[i].y_velocity +
          //                                    env->targets[i].z_velocity * env->targets[i].z_velocity));
          // printf("Target singer_sigma: %f\n", env->targets[i].singer_sigma);
          // printf("Target singer_theta: %f\n", env->targets[i].singer_theta);
          // printf("Target priority: %f\n", env->targets[i].priority);
          env->targets[i].is_tracked = false;
          env->rewards[0] -= TRACK_LOSS_PENALTY;
        }
        // t_dwell_estimate does not change
      }
    }

    // Update locations

    for (int i = 0; i < env->max_trackers; i++) {
      env->targets[i].x +=
          env->targets[i].x_velocity * delta_t / 1000.0f +
          env->targets[i].x_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].y +=
          env->targets[i].y_velocity * delta_t / 1000.0f +
          env->targets[i].y_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].z +=
          env->targets[i].z_velocity * delta_t / 1000.0f +
          env->targets[i].z_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].x_velocity += env->targets[i].x_acceleration * delta_t / 1000.0f;
      env->targets[i].y_velocity += env->targets[i].y_acceleration * delta_t / 1000.0f;
      env->targets[i].z_velocity += env->targets[i].z_acceleration * delta_t / 1000.0f;
      float rho = exp(-(delta_t / 1000.0f) / env->targets[i].singer_theta);
      env->targets[i].x_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma);
      env->targets[i].y_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma);
      env->targets[i].z_acceleration +=
          sqrt(1 - rho * rho) * normal(0, env->targets[i].singer_sigma) * VERTICAL_MOTION_FACTOR;
    }

    // Reset targets that have gone out of bounds
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].x < 0 || env->targets[i].x > MAX_TARGET_XY_RANGE ||
          env->targets[i].y < 0 || env->targets[i].y > MAX_TARGET_XY_RANGE ||
          env->targets[i].z < 0 || env->targets[i].z > MAX_TARGET_Z_RANGE ||
          sqrt(env->targets[i].x * env->targets[i].x + env->targets[i].y * env->targets[i].y +
                  env->targets[i].z * env->targets[i].z) >
              S_BAND_MAX_RANGE) {
        initialize_target(env, i);
      }
    }

    if (env->tick > 60000)  // 1 minute ?
    {
      env->terminals[0] = 1;
      env->rewards[0] = -1.0;
      c_reset(env);
      return;
    }
  }
}

// Required function. Should handle creating the client on first call
void c_render(Radarxs* env) {
// The plan position indicator will be square on left of screen
  // Origin will be at bottom-left
  // X axis positive to right
  // Y axis positive to top
  // I'm going to initially just try to do a WINDOW_Y_PX x WINDOW_Y_PX ppi with
  // a 240 x 80 search indicator on the top right and just hope they don't
  // overlap. int ppi_dimension = WINDOW_Y_PX * scale; int
  // search_indicator_width = 8 * scale;

  int scale = 2;

  if (!IsWindowReady()) {
    if (!(scale == 1 || scale == 2 || scale == 4 || scale == 8)) {
      fprintf(stderr, "Error: scale is one of 1,2,4,8. (4 is 1080p, 8 is 4k).\n");
      exit(1);
    }
    InitWindow(WINDOW_X_PX * scale, WINDOW_Y_PX * scale, "PufferLib Radars");
    SetTargetFPS(10);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(color_bgteal);

  // Draw the search indicator
  int cell_width = 9 * scale;
  int cell_height = 9 * scale;
  int grid_x = WINDOW_X_PX * scale - MAX_AZ_SLICES * cell_width;

  for (int i = 0; i < MAX_EL_SLICES; i++) {
    for (int j = 0; j < MAX_AZ_SLICES; j++) {
      int sector = i * MAX_AZ_SLICES + j;
      int zero_cost_time_remaining = env->observations[sector];
      Color color;
      if (zero_cost_time_remaining >= ZERO_COST_SEARCH_TIME - SEARCH_DWELL_TIME) {
        color = color_red;
      } else if (zero_cost_time_remaining > 0) {
        // Map zero_cost_time_remaining to a cyan - to - grey scale
        float good_intensity = (float)zero_cost_time_remaining / ZERO_COST_SEARCH_TIME;
        if (good_intensity < 0) {
          good_intensity = 0;
        }
        if (good_intensity > 1) {
          good_intensity = 1;
        }
        color = Fade(color_cyan, good_intensity);

      } else if (zero_cost_time_remaining < 0) {
        // Map zero_cost_time_remaining to a yellow - to - red scale
        float red_intensity = (float)abs(zero_cost_time_remaining) / ZERO_COST_SEARCH_TIME;
        if (red_intensity < 0) {
          red_intensity = 0;
        }
        if (red_intensity > 1) {
          red_intensity = 1;
        }
        color = (Color){(unsigned char)(255), (unsigned char)(255 - red_intensity * 255),
                        (unsigned char)(0), 255};
      } else {
        color = GRAY;
      }
      DrawRectangle(grid_x + j * cell_width, i * cell_height, cell_width, cell_height, color);
    }
  }

  // Draw the sensor ranges

  float ppmm = WINDOW_Y_PX * scale / S_BAND_MAX_RANGE;
  Vector2 center = {0, WINDOW_Y_PX * scale};

  // DrawRingLines(center, innerRadius, outerRadius, startAngle, endAngle,
  // (int)segments, color)
  DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                color_gray);
  if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] ==
      S_BAND_SENSOR) {
    DrawRingLines(center, X_BAND_MIN_RANGE * ppmm, X_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_gray);
    DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_yellow);

  } else {
    DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_gray);
    DrawRingLines(center, X_BAND_MIN_RANGE * ppmm, X_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_yellow);
  }

  int action = env->actions[0];
  if (action == SEARCH) {
    DrawText("SEARCH", 130 * scale, 5 * scale, 10 * scale, WHITE);
  } else if (action <= env->max_trackers) {
    DrawText(TextFormat("TRACK %d", action), 130 * scale, 5 * scale,
             10 * scale, WHITE);
    action -= 1;
    // if (env->targets[action].is_tracked) {
    //   DrawCircle(env->targets[action].x * ppmm,
    //              WINDOW_Y_PX * scale - env->targets[action].y * ppmm, 10 * scale,
    //              color_red);
    // }
  }

  

  DrawText(TextFormat("TIME: %d ms", env->tick), 130 * scale, 15 * scale,
           10 * scale, WHITE);

  // Print the reward, print the reward per step
  DrawText(TextFormat("REWARD: %.2f", env->rewards[0]), 280 * scale, 100 * scale,
          10 * scale, WHITE);
  DrawText(TextFormat("SUM REWARD: %.2f", env->log.episode_return), 280 * scale, 110 * scale,
           10 * scale, WHITE);
  DrawText(TextFormat("REWARD/SECOND: %.4f", env->log.episode_return / env->tick * 1000), 280 * scale,
           130 * scale, 10 * scale, WHITE);
  
  // The time remaining for each sensor
  DrawText(TextFormat("S-BAND BUSY: %d ms", env->s_band_t_until_free), 280 * scale, 140 * scale,
           10 * scale, WHITE);
  DrawText(TextFormat("X-BAND BUSY: %d ms", env->x_band_t_until_free), 280 * scale, 150 * scale,
           10 * scale, WHITE);

  // Draw the targets
  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active) {
      float x = env->targets[i].x * ppmm;
      float y = env->targets[i].y * ppmm;
      Vector2 position = {x, WINDOW_Y_PX * scale - y};
      float heading = atan2f(env->targets[i].y_velocity, env->targets[i].x_velocity);
      float arrow_size = 2 * scale;
      if (i == action) {
        arrow_size = 3 * scale;
      }
      Vector2 points[3] = {
          (Vector2){position.x + arrow_size * cosf(heading + PI / 2),
                    position.y + arrow_size * sinf(heading + PI / 2)},
          (Vector2){position.x + 3 * arrow_size * cosf(heading),
                    position.y + 3 * arrow_size * sinf(heading)},
          (Vector2){position.x + arrow_size * cosf(heading - PI / 2),
                    position.y + arrow_size * sinf(heading - PI / 2)},
      };
      if (env->targets[i].is_tracked) {
        if (i == action) {
          DrawTriangle(points[0], points[1], points[2], color_red);
        } else if (env->targets[i].priority >= 2) {
          DrawTriangle(points[0], points[1], points[2], color_blue);
        } else if (env->targets[i].priority >= 1) {
          DrawTriangle(points[0], points[1], points[2], color_sky);
        } else {
          DrawTriangle(points[0], points[1], points[2], color_cyan);
        }
      } else {
        // This draws untracked targets
        // DrawTriangle(points[0], points[1], points[2], color_silver);
      }
    }
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Radarxs* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
