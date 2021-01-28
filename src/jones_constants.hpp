#include <cmath>

#ifndef MINDS_CRAWL_JONES_CONSTANTS_HPP
#define MINDS_CRAWL_JONES_CONSTANTS_HPP

/**
 * A namespace containing model-level constants for simulation
 *
 * @note Constants themselves are not documented yet, see https://www.overleaf.com/read/vnwxqhdhzknw for details
 */
namespace jones_constants
{
    /// How far do particle sensors stay from an agent (`so` stays for sensor offset)
    const double so = 7;

    /// The angle between left and middle, middle and right particle sensors (`sa` stays for sensor angle)
    const double sa = M_PI * 45 / 180;

    /// The angle at which particle rotates when needs it (rotation based on sensor values or random rotation)
    const double ra = M_PI * 45 / 100;


    /**
     * Oscillatory mode (on/off)
     *
     * When particle A is trying to move to a space point where it must be reattached to another `MapNode`, but the
     * destination node contains a particle already, the first particle doesn't move. If oscillatory mode is off, first
     * particle also chooses a random direction, when it's on the particle just stays on it's place and tries to move
     * forward again on next iteration
     */
    const bool osc = false;

    /**
     * Probability of random direction change (`pcd` most likely stays for probability change direction)
     *
     * On each iteration every particle randomly changes it's direction with this probability
     */
    const double pcd = 0;

    /**
     * How many trail is added after a particle successfully moves (unsuccessful move is when we are trying to
     * reattach to a node which has a particle attached already
     */
    const double dept = 5;

    /**
     * Step size of a particle. How far does particle move on each step
     *
     * @warning The `speed` value <b>MUST</b> be <b>AT MOST</b> half of <b>minimal</b> distance between two `MapNode`s
     */
    const double speed = 0.2;


    /// Whether we have to run random death test
    const bool do_random_death_test = false;

    /**
     * How often (in number of iterations) to run division test
     *
     * @see division_test
     */
    const int division_test_frequency = 5;

    /**
     * How often (in number of iterations) to run death test
     *
     * @see death_test
     */
    const int death_test_frequency = 5;

    /**
     * The probability that a particle will be actually divided if it satisfies division conditions
     *
     * @see division_test
     */
    const double division_probability = 1;

    /**
     * The probability that particle dies during a random death test
     *
     * @see random_death_test
     */
    const double random_death_probability = 0;


    /**
     * The size of a node window being considered in division test (`gw` most likely stays for growth window)
     *
     * @see division_test
     */
    const int gw = 9;

    /**
     * The minimum value of particles must be in a growth window to continue division
     * (`gmin` most likely stays for growth minimum)
     *
     * @see division_test
     */
    const int gmin = 0;

    /**
     * The maximum value of particles must be in a growth window to continue division
     * (`gmax` most likely stays for growth maximum)
     *
     * @see division_test
     */
    const int gmax = 10;

    /**
     * The size of a node window being considered in death test (`sw` most likely stays for survival window)
     *
     * @see death_test
     */
    const int sw = 5;

    /**
     * The minimum value of particles must be in a survival window to leave a particle alive
     * (`smin` most likely stays for survival minimum)
     *
     * @see death_test
     */
    const int smin = 0;

    /**
     * The maximum value of particles must be in a survival window to leave a particle alive
     * (`smax` most likely stays for survival maximum)
     *
     * @see death_test
     */
    const int smax = 24;


    /**
     * A value such that new trail value is an average trail in a 3x3 node window multiplied by `(1 - diffdamp)
     *
     * @see diffuse_trail
     */
    const double diffdamp = 0.1;

    /// How much trail to add to a node if there is food in it (if there are no particle in a 3x3 node window)
    const double projectvalue = 2;

    /// How much trail to add to a node if there is food in it (if there is at least one particle in a 3x3 node window)
    const double suppressvalue = 2.5;

    /// Whether we're projecting food to trail (= whether particles see food)
    const bool projectnutrients = true;

    /**
     *  Starting from which iteration to start projecting food to trail (= starting from which iteration particles
     *  start seeing food)
     */
    const int startprojecttime = 0;
}

#endif //MINDS_CRAWL_JONES_CONSTANTS_HPP
