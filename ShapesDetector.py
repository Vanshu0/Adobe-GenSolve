import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.spatial import distance
import svgwrite
import cairosvg
from sklearn.metrics import r2_score
from scipy.optimize import least_squares
from numpy.linalg import inv


class Curvetopia:
    """
    A class to handle reading paths from a CSV, processing and regularizing those paths,
    and plotting or saving them to an SVG file.

    Attributes:
    ----------
    csv_path : str
        Path to the input CSV file containing paths.
    svg_path : str
        Path to the output SVG file.
    paths_XYs : list
        List of paths read from the CSV file.
    colours : list
        List of colours used for plotting different paths.
    """

    def __init__(self, csv_path, svg_path):
        """
        Initializes the Curvetopia class with the CSV path and SVG path.

        Parameters:
        ----------
        csv_path : str
            Path to the input CSV file.
        svg_path : str
            Path to the output SVG file.
        """
        self.csv_path = csv_path
        self.svg_path = svg_path
        self.paths_XYs = self.read_csv()
        self.colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    def read_csv(self):
        """
        Reads the CSV file and parses it into paths.

        Returns:
        -------
        list
            A list of paths, where each path is a list of 2D numpy arrays.
        """
        np_path_XYs = np.genfromtxt(self.csv_path, delimiter=',')
        path_XYs = []

        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []

            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY)

            path_XYs.append(XYs)

        return path_XYs

    def plot(self):
        """
        Plots the paths using Matplotlib.
        """
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))

        for i, XYs in enumerate(self.paths_XYs):
            c = self.colours[i % len(self.colours)]
            for XY in XYs:
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)

        ax.set_aspect('equal')
        plt.show()

    def polylines2svg(self):
        """
        Converts the paths to SVG format and saves the SVG and PNG files.
        """
        W, H = 0, 0

        for path_XYs in self.paths_XYs:
            for XY in path_XYs:
                W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))

        padding = 0.1
        W, H = int(W + padding * W), int(H + padding * H)
        dwg = svgwrite.Drawing(self.svg_path, profile='tiny', shape_rendering='crispEdges')
        group = dwg.g()

        for i, path in enumerate(self.paths_XYs):
            path_data = []
            c = self.colours[i % len(self.colours)]
            for XY in path:
                path_data.append(("M", (XY[0, 0], XY[0, 1])))
                for j in range(1, len(XY)):
                    path_data.append(("L", (XY[j, 0], XY[j, 1])))
                if not np.allclose(XY[0], XY[-1]):
                    path_data.append(("Z", None))
            group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
        dwg.add(group)
        dwg.save()
        png_path = self.svg_path.replace('.svg', '.png')
        fact = max(1, 1024 // min(H, W))
        cairosvg.svg2png(url=self.svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W,
                         output_height=fact * H, background_color='white')

    @staticmethod
    def regularize_straight_lines(points):
        """
        Regularizes a set of points into a straight line using linear regression.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form a straight line.
        """
        model = LinearRegression()
        model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
        y_pred = model.predict(points[:, 0].reshape(-1, 1))
        return np.column_stack((points[:, 0], y_pred))

    @staticmethod
    def regularize_circles(points):
        """
        Regularizes a set of points into a circle.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form a circle.
        """
        center = points.mean(axis=0)
        radii = np.linalg.norm(points - center, axis=1)
        radius = np.mean(radii)
        angles = np.linspace(0, 2 * np.pi, len(points))
        circle_points = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
        return circle_points

    @staticmethod
    def regularize_ellipses(points):
        """
        Regularizes a set of points into an ellipse.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form an ellipse.
        """
        kmeans = KMeans(n_clusters=2).fit(points)
        centers = kmeans.cluster_centers_
        center = points.mean(axis=0)
        major_axis = np.linalg.norm(centers[0] - centers[1])
        minor_axis = np.mean([distance.euclidean(point, center) for point in points])
        angles = np.linspace(0, 2 * np.pi, len(points))
        ellipse_points = np.column_stack((center[0] + major_axis * np.cos(angles) / 2, center[1] + minor_axis * np.sin(angles) / 2))
        return ellipse_points

    @staticmethod
    def regularize_rectangles(points):
        """
        Regularizes a set of points into a rectangle.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form a rectangle.
        """
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        rectangle_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]])
        return rectangle_points

    @staticmethod
    def regularize_polygons(points, sides=4):
        """
        Regularizes a set of points into a polygon with a given number of sides.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.
        sides : int, optional
            Number of sides of the polygon (default is 4).

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form a polygon.
        """
        center = points.mean(axis=0)
        angles = np.linspace(0, 2 * np.pi, sides + 1)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        polygon_points = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
        return polygon_points

    @staticmethod
    def regularize_stars(points):
        """
        Regularizes a set of points into a star shape.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        numpy.ndarray
            2D array of points adjusted to form a star shape.
        """
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        star_points = []
        for i in range(len(sorted_points)):
            star_points.append(sorted_points[i])
            star_points.append(center + (center - sorted_points[i]) / 2)
        return np.array(star_points)

    def main(self):
        """
        Main function to process and regularize paths, plot them, and save them to SVG.
        """
        regularized_paths = []

        for XYs in self.paths_XYs:
            regularized_XYs = []
            for XY in XYs:
                if self.is_straight_line(XY):
                    regularized_XYs.append(self.regularize_straight_lines(XY))
                elif self.is_circle(XY):
                    regularized_XYs.append(self.regularize_circles(XY))
                elif self.is_ellipse(XY):
                    regularized_XYs.append(self.regularize_ellipses(XY))
                elif self.is_rectangle(XY):
                    regularized_XYs.append(self.regularize_rectangles(XY))
                elif self.is_polygon(XY):
                    regularized_XYs.append(self.regularize_polygons(XY))
                elif self.is_star(XY):
                    regularized_XYs.append(self.regularize_stars(XY))
                else:
                    regularized_XYs.append(XY)
            regularized_paths.append(regularized_XYs)

        self.paths_XYs = regularized_paths
        self.plot()
        self.polylines2svg()

    @staticmethod
    def is_straight_line(points, tolerance=0.99):
        """
        Detects if the given points form a straight line.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.
        tolerance : float, optional
            R^2 score threshold to consider points as forming a straight line (default is 0.99).

        Returns:
        -------
        bool
            True if points form a straight line, False otherwise.
        """
        if points.shape[0] < 2:
            return False

        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        return r2 >= tolerance

    @staticmethod
    def is_circle(points, tolerance=0.1):
        """
        Detects if the given points form a circle.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.
        tolerance : float, optional
            Mean squared error threshold to consider points as forming a circle (default is 0.1).

        Returns:
        -------
        bool
            True if points form a circle, False otherwise.
        """
        if points.shape[0] < 3:
            return False

        def calc_R(xc, yc):
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

        def f(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(points, axis=0)
        result = least_squares(f, center_estimate)

        center = result.x
        radius = np.mean(calc_R(*center))
        residuals = f(center)
        mse = np.mean(residuals ** 2)

        return mse < tolerance

    @staticmethod
    def is_ellipse(points, tolerance=0.1):
        """
        Detects if the given points form an ellipse.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.
        tolerance : float, optional
            Mean squared error threshold to consider points as forming an ellipse (default is 0.1).

        Returns:
        -------
        bool
            True if points form an ellipse, False otherwise.
        """
        if points.shape[0] < 5:
            return False

        D = np.column_stack((points[:, 0] ** 2, points[:, 0] * points[:, 1], points[:, 1] ** 2, points[:, 0], points[:, 1],
                             np.ones(points.shape[0])))

        S = np.dot(D.T, D)

        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        eigvals, eigvecs = np.linalg.eig(np.dot(inv(S), C))

        eigvec = eigvecs[:, np.argmax(eigvals)]

        a, b, c, d, e, f = eigvec

        def ellipse_dist(x, y):
            return a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f

        distances = np.abs(ellipse_dist(points[:, 0], points[:, 1]))
        mse = np.mean(distances ** 2)

        return mse < tolerance

    @staticmethod
    def is_rectangle(points):
        """
        Placeholder for detecting if points form a rectangle.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        bool
            True if points form a rectangle, False otherwise.
        """
        pass

    @staticmethod
    def is_polygon(points):
        """
        Placeholder for detecting if points form a regular polygon.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        bool
            True if points form a polygon, False otherwise.
        """
        pass

    @staticmethod
    def is_star(points):
        """
        Placeholder for detecting if points form a star shape.

        Parameters:
        ----------
        points : numpy.ndarray
            2D array of points.

        Returns:
        -------
        bool
            True if points form a star shape, False otherwise.
        """
        pass


# Example usage
csv_path = r'problems\isolated.csv'
svg_path = r'output_file.svg'
curvetopia = Curvetopia(csv_path, svg_path)
curvetopia.main()
