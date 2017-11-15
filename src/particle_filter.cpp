#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    // x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    num_particles  = 100;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    default_random_engine gen;
    for (int i=0; i<num_particles; ++i)
    {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    default_random_engine gen;
    for (int i=0; i<num_particles; ++i)
    {
        // Check if yaw rate is 'zero'
        if(fabs(yaw_rate)<0.00001)
        {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else
        {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        }
        // Add noise to the particles
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the
    // observed measurement to this particular landmark.
    for (int i=0; i<observations.size(); ++i)
    {
        // Init minimum distance
        double min_dist = numeric_limits<double>::max();
        // Init landmark id
        int map_id = -1;
        for (int j=0; j<predicted.size(); ++j)
        {
            // Calc distance between current and predicted landmarks
            double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            // Find the predicted landmark nearest the current observed landmark
            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                map_id = predicted[j].id;
            }
        }
        // Assign the observed measurement to this particular landmark
        observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution.
    for (int i=0; i<num_particles; ++i)
    {
      // Get the particle x, y coordinates
      double px = particles[i].x;
      double py = particles[i].y;
      double ptheta = particles[i].theta;
      vector<LandmarkObs> predictions;

      // For each map landmark
      for (int j=0; j<map_landmarks.landmark_list.size(); ++j)
      {
        // Get id and x,y coordinates
        float lx = map_landmarks.landmark_list[j].x_f;
        float ly = map_landmarks.landmark_list[j].y_f;

        // Consider landmarks within sensor range of the particle
        float x_diff = fabs(lx - px);
        float y_diff = fabs(ly - py);
        if (x_diff <= sensor_range && y_diff <= sensor_range)
        {
          // Add prediction
          predictions.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, lx, ly });
        }
      }

      // List of transformed observations from vehicle coordinates to map coordinates
      vector<LandmarkObs> transformed_obs;
      for (int j=0; j<observations.size(); ++j)
      {
        double tx = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
        double ty = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
        transformed_obs.push_back(LandmarkObs{observations[j].id, tx, ty });
      }

      // Data association for the predictions and transformed observations
      dataAssociation(predictions, transformed_obs);
      // Re-init weight
      particles[i].weight = 1.0;
      for (int j=0; j<transformed_obs.size(); ++j)
      {
        double x, y;
        double mu_x = transformed_obs[j].x;
        double mu_y = transformed_obs[j].y;
        // Get x,y coordinates of the prediction associated with the current observation
        for (int k=0; k<predictions.size(); ++k)
        {
          if (predictions[k].id == transformed_obs[j].id)
          {
            x = predictions[k].x;
            y = predictions[k].y;
          }
        }
        // Calculate weight for this observation with multivariate Gaussian
        double sigx = std_landmark[0];
        double sigy = std_landmark[1];
        double gauss_norm =  1/(2 * M_PI * sigx * sigy);
        double exponent  = (pow(x-mu_x, 2) / (2 * sigx * sigx)) + (pow(y-mu_y, 2) / (2 * sigy * sigy));
        double obs_weight = gauss_norm * exp(-exponent);
        // Product of this obersvation weight with total observations weight
        particles[i].weight *= obs_weight;
      }
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    vector<Particle> new_particles;
    default_random_engine gen;
    // Get all of the current weights
    //vector<double> weights(num_particles);
    //generate(weights.begin(), weights.end(), [i=0, this] () mutable {return particles[i++].weight;});
    vector<double> weights;
    for (int i=0; i<num_particles; ++i)
    {
        weights.push_back(particles[i].weight);
    }
    // Get max weight
    double max_weight = *max_element(weights.begin(), weights.end());
    // Generate random starting index for resampling wheel
    uniform_int_distribution<int> uni_dist(0, num_particles-1);
    uniform_real_distribution<double> uni_real_dist(0.0, max_weight);
    int index = uni_dist(gen);
    double beta = 0.0;
    // Resampling wheel
    for (int i=0; i<num_particles; ++i)
    {
        beta += uni_real_dist(gen) * 2.0;
        while (weights[index] < beta)
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
     }
     particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
