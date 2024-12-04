import numpy as np

def get_observation_point_locations(icemask, usurf, covered_area):
    gx, gy = np.where(icemask)
    glacier_points = np.array(list(zip(gx, gy)))
    num_sample_points = int((covered_area / 100) * np.sum(icemask))
    print('Number of points: {}'.format(num_sample_points))

    random_state = np.random.RandomState(seed=420)
    observation_index = random_state.choice(len(glacier_points),
                                            num_sample_points, replace=False)
    observation_points = glacier_points[observation_index]

    def get_pixel_value(point):
        x, y = point
        return usurf[x][y]

    sorted_observation_points = sorted(observation_points, key=get_pixel_value)
    observation_points = np.array(sorted_observation_points)

    return observation_points


def get_observation_point_values(EnKF_object, points):
    """
    Calculate observable values for a list of points based on the surface height differences.

    Parameters:
        EnKF_object: An object containing `ensemble_usurf_log` and `observation_point_location`.
        points (list of tuples): A list of (x, y) coordinates representing the points of interest.

    Returns:
        dict: A dictionary with 'mean_dhdt' and per-point dhdt values as arrays with shape (ensemble, iter-1).
    """
    ensemble_usurf_log = EnKF_object.ensemble_usurf_log

    # Check that there are at least two time steps
    if len(ensemble_usurf_log) < 2:
        return {'mean_dhdt': None, 'points_dhdt': None}

    # Initialize an empty list to store the differences
    dhdt_ensemble_all = []

    # Loop through all consecutive time steps
    for i in range(1, len(ensemble_usurf_log)):
        current_usurf_ensemble = np.array(ensemble_usurf_log[i])
        previous_usurf_ensemble = np.array(ensemble_usurf_log[i - 1])

        # Compute the difference between the current and previous time step
        dhdt_ensemble = current_usurf_ensemble - previous_usurf_ensemble
        dhdt_ensemble_all.append(dhdt_ensemble)

    # Convert the differences to a numpy array
    dhdt_ensemble_all = np.array(
        dhdt_ensemble_all)  # Shape: (iter-1, ensemble, x, y)

    # Compute mean dh/dt across the spatial dimensions (x, y)
    mean_dhdt = dhdt_ensemble_all.mean(axis=(2, 3))  # Shape: (iter-1, ensemble)

    # Compute dhdt values for the given points and create dictionary entries
    point_observables = {}
    for i, (x, y) in enumerate(points):
        point_key = f"point_{i + 1}_dhdt"  # Create a key like 'point_1_dhdt', 'point_2_dhdt', etc.
        point_dhdt = dhdt_ensemble_all[:, :, x, y]  # Shape: (iter-1, ensemble)
        point_observables[point_key] = point_dhdt.T  # Shape: (ensemble, iter-1)

    # Combine mean_dhdt and point observables into a single dictionary
    observables = {
        'mean_dhdt': mean_dhdt.T,  # Shape: (ensemble, iter-1),
        **point_observables  # Add point-specific entries dynamically
    }

    return observables

