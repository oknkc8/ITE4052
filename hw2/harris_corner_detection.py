import numpy as np
import cv2
import time
import argparse
from tqdm import tqdm

def detect_corner(img, window_size, method):
	H, W = img.shape
	corner_img = np.zeros((H,W))

	# Compute Gradients
	dy, dx = np.gradient(img)
	Ix2 = dx * dx
	Ixy = dy * dx
	Iy2 = dy * dy

	offset = window_size // 2

	for y in tqdm(range(offset, H-offset)):
		for x in range(offset, W-offset):
			A = np.sum(Ix2[y-offset : y+offset+1, x-offset : x+offset+1])
			B = np.sum(Ixy[y-offset : y+offset+1, x-offset : x+offset+1])
			C = np.sum(Iy2[y-offset : y+offset+1, x-offset : x+offset+1])

			H = np.array([[A, B],
						  [B, C]])
			
			# Compute Eigenvalues
			lambdas = np.linalg.eigvals(H)
			
			if lambdas[0] + lambdas[1] == 0:
				continue

			# Get Harris Response
			if method == 'Shi-Tomasi':
				if np.min(lambdas) != 0:
					corner_img[y, x] = np.min(lambdas)
			else:
				f = (lambdas[0] * lambdas[1]) / (lambdas[0] + lambdas[1])
				if f > 0.04:
					corner_img[y, x] = f

	cv2.normalize(corner_img, corner_img, 0.0, 255.0, cv2.NORM_MINMAX)

	return corner_img


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--window_size', type=int, default=3, help='Window Size')
	parser.add_argument('--method', type=str, default='Shi-Tomasi', help='Detect Method - Shi-Tomasi & Harris-Response')
	args = parser.parse_args()

	
	# Read Checkerboard Image
	img = cv2.imread('input/checkerboard.png', cv2.IMREAD_GRAYSCALE)
	img = (img / 255).astype(np.float32)

	# Detect Corner
	end = time.time()
	result = detect_corner(img, args.window_size, args.method)
	running_time = time.time() - end

	cv2.imwrite("result/result_{}_{}.png".format(args.method, args.window_size), result)
	print("Time: {} (Window Size : {})".format(running_time, args.window_size))