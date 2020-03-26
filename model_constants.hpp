#ifndef MIND_S_CRAWL_MODEL_CONSTANTS_HPP
#define MIND_S_CRAWL_MODEL_CONSTANTS_HPP

/* PLEASE REMEMBER TO UPDATE THESE VALUES BEFORE RUNNING IN 3D
 * THIS IS EXTREMELY FUCKING IMPORTANT
 */

namespace jones_constants
{
    // See https://www.overleaf.com/read/vnwxqhdhzknw for parameters' descriptions

    const bool osc = false;
    const double pcd = 0;
    const double oscresetprob = 0.01;
    const double dept = 5;

    const int popsize = 16'000;
    const bool do_random_death_test = false;
    const int division_frequency_test = 5;
    const int death_frequency_test = 5;
    const double division_probability = 1;
    const double death_random_probability = 0;
    const int gw = 9;
    const int gmin = 0;
    const int gmax = 10;
    const int sw = 5;
    const int smin = 0;
    const int smax = 24;
    const int divisionborder = 5;

    const double diffdamp = 0.1;
    const double projectvalue = 2;
    const double suppressvalue = 2.5;
    const bool projectnutrients = true;
    const int startprojecttime = 0;
}

#endif //MIND_S_CRAWL_MODEL_CONSTANTS_HPP
