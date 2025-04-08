from train_env import LearmArmEnv
import numpy as np
import time
import os
import csv

def run_heuristic_trial(env, visualize=False):
    obs, _ = env.reset()
    ball_pos, net_pos, xy_dist = None, None, None

    caught = False
    for _ in range(env.max_ep_steps + 0):
        ball_x, ball_y, _, _ = obs

        # Heuristic mapping: assume base angle tracks y, shoulder tracks x
        base_target = np.clip((ball_y / 3.5) * -1.57, -1.57, 1.57)
        shoulder_target = np.clip((ball_x / 3.5) * 1.57, 0, 1.57)

        # Apply POSITION_CONTROL
        env.step_counter += 1
        pyB = env._get_pybullet()  # Hacky, but fine for now
        pyB.setJointMotorControl2(env.robot_id, env.base_joint_idx, pyB.POSITION_CONTROL, targetPosition=base_target)
        pyB.setJointMotorControl2(env.robot_id, env.shoulder_joint_idx, pyB.POSITION_CONTROL, targetPosition=shoulder_target)
        pyB.setJointMotorControl2(env.robot_id, env.wrist_joint_idx, pyB.POSITION_CONTROL, targetPosition=shoulder_target)

        pyB.stepSimulation()
        if visualize:
            time.sleep(env.time_step)

        # Check catch: Is ball center close to net center when it has similar z-elevation as the net?
        ball_pos, _ = pyB.getBasePositionAndOrientation(env.ball_id)[:2]  # Get the position of the ball (no need for orientation)
        net_pos, _ = pyB.getLinkState(env.robot_id, env.net_joint_idx)[:2]  # Get the position of the net link (no need for orientation)
        xy_dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(net_pos[:2]))
        if abs(ball_pos[2] - net_pos[2]) <= 0.1 and xy_dist <= 0.55:  # ball center is within 55mm from net center
            caught = True
            break

        obs = env._get_obs()

    # ball_xy = tuple(round(coord, 1) for coord in ball_pos[:2])
    # net_xy = tuple(round(coord, 1) for coord in net_pos[:2])
    # print("ball_xy:", ball_xy, "| net_xy:", net_xy, "| dist:", round(xy_dist, 1))
    return caught


def main():
    num_trials = 1000
    visualize = True if num_trials <= 20 else False
    env = LearmArmEnv(render=visualize)
    success_count = 0
    results = []

    for i in range(num_trials):
        caught = run_heuristic_trial(env, visualize=visualize)
        success_count += int(caught)
        results.append([i+1, success_count])
        print(f"Trial {i+1}: {'Caught ✅' if caught else 'Missed ❌'}")
        
    print(f"\nHeuristic success rate: {success_count} / {num_trials}")

    # Save results to a CSV file in the "results" folder
    if not os.path.exists("results"):
        os.makedirs("results")  # Create the folder if it doesn't exist
    
    with open('results/heuristic_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial', 'Cumulative Success Count'])  # Header row
        writer.writerows(results)  # Write the results for each trial
    
    env.close()

if __name__ == "__main__":
    main()
