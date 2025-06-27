import pickle
import argparse
import os
import subprocess
import numpy as np
import open3d as o3d
import copy

from deconstruct.utils.open3d_utils.general_utils import get_absolute_path_to_repo, numpy_arr_to_o3d


def intensify_rgb_colors(color, exponent=2.5):
    """
    this intensifies the color by taking the color to the power of exponent.
    Using exponent=2.5 empirically makes the rgb colors in open3d visualization more faithful.
    """
    color = color ** exponent
    return color


class VisualizeObject:
    # class attributes go here:
    mesh_pickle_attributes = ["vertex_colors", "vertices", "triangles"]
    points_pickle_attributes = ["points", "normals", "colors"]
    lines_pickle_attributes = ["points", "lines", "colors"]

    def __init__(self, geometry, name=None, is_visible=True, intensify_colors=True):
        """
        Class used for visualization of o3d.geometry.Geometry objects.

        Parameters
        ----------
        geometry : o3d.geometry.Geometry
        name : str, optional
            Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.
        """

        assert isinstance(geometry, o3d.geometry.Geometry)
        self.geometry = copy.deepcopy(geometry)  # useful if colors are intensified
        self.name = name
        self.pickle_attributes = None  # to be specified in subclasses
        self.is_visible = is_visible

        # intensify colors to make them more faithful rgb colors:
        if intensify_colors:
            color_attr = None
            if hasattr(self.geometry, "colors"):
                color_attr = "colors"
            elif hasattr(self.geometry, "vertex_colors"):
                color_attr = "vertex_colors"

            if color_attr is not None:
                colors = np.asarray(getattr(self.geometry, color_attr))
                colors = intensify_rgb_colors(colors)
                setattr(self.geometry, color_attr, o3d.utility.Vector3dVector(colors))

    def __getstate__(self):
        """
        Make class pickleable.
        Makes the VisualizeObject pickleable (o3d.geometry.Geometry can not be pickled otherwise).
        This is not the clean way to do this. See e.g.
        https://stackoverflow.com/questions/1939058/simple-example-of-use-of-setstate-and-getstate
        Or part_class.py.

        Returns
        -------

        """
        assert hasattr(self, "pickle_attributes"), "objects need to have a list of pickle_attributes"
        geometry_attr = {"type": type(self.geometry)}
        for attr in self.pickle_attributes:
            geometry_attr[attr] = np.asarray(getattr(self.geometry, attr))

        self.geometry_attr = geometry_attr
        self_dict = self.__dict__.copy()
        del self_dict["geometry"]

        return self_dict

    def __setstate__(self, d):
        """
        Reconstruct class after pickling
        To reconstruct the o3d.geometry.Geometry after pickling.

        Returns
        -------

        """
        self.__dict__ = d
        assert hasattr(self, "geometry_attr"), "geometry_attr needed for reconstruction was not found."
        # instantiate object by type:
        geometry = self.geometry_attr["type"]()
        for attr in self.pickle_attributes:
            # print("attr=", attr, numpy_arr_to_o3d(self.geometry_attr[attr]))
            setattr(geometry, attr, numpy_arr_to_o3d(self.geometry_attr[attr]))

        self.geometry = geometry


class VisualizeMesh(VisualizeObject):

    def __init__(self, show_wire_frame=True, wire_frame_width=5, wire_frame_color="default",
                 light_reflections=True, alpha=1., flip_xyz=False, **super_kwargs):
        """
        Class used for visualization of o3d.geometry.TriangleMesh objects.

        Parameters
        ----------
        show_wire_frame : bool, optional
            Specifies if mesh wire frame is toggled on or off. Default = True.
        wire_frame_width : float, optional
            Line width of wire frame. Default = 5
        wire_frame_color : tuple or np.ndarray or str, optional
            Color of wire frame. Default =  np.zeros(3)
        light_reflections : bool, optional
            Specify if light reflections are present. Default = True.
        alpha : float, optional
            Between [0,1]. Specifies transparency of mesh.
        geometry : o3d.geometry.Geometry
            Geometry object of open3d library.
        name : str, optional
            Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.
        super_kwargs
            geometry : o3d.geometry.Geometry
            name : str, optional
                Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
            is_visible : bool, optional
                Specify if geometry should be toggled on or off in viewer. Default = True.
        """

        # call superclass
        super().__init__(**super_kwargs)
        assert isinstance(self.geometry,
                          o3d.geometry.TriangleMesh), "geometry must be instance of o3d.geometry.TriangleMesh"

        self.show_wire_frame = show_wire_frame
        self.wire_frame_width = wire_frame_width
        self.wire_frame_color = np.zeros(3) if wire_frame_color == "default" else wire_frame_color
        self.light_reflections = light_reflections
        self.alpha = alpha
        self.flip_xyz = flip_xyz
        self.pickle_attributes = self.mesh_pickle_attributes

        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1."
        if show_wire_frame and alpha < 1.:
            print("Effect of alpha transparency really only works with show_wire_frame=False.")

    def flip_xyz_to_zyx(self):
        vertices = np.asarray(self.geometry.vertices)
        vertices = np.flip(vertices, axis=1)
        self.geometry.vertices = o3d.utility.Vector3dVector(vertices)

    def create_visualize_dict_list(self):
        if self.flip_xyz:
            self.flip_xyz_to_zyx()

        if self.light_reflections:
            # vertex normals required for shading
            self.geometry.compute_vertex_normals()

        visualize_dict = {"geometry": self.geometry, "is_visible": self.is_visible}
        if self.name is not None:
            visualize_dict["name"] = self.name
        if self.alpha < 1:
            # set transparency:
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.base_color = [0.5, 0.5, 0.5, self.alpha]  # first 3 entries are overwritten if mesh has vertex colors.
            mat.shader = 'defaultLitTransparency'
            mat.base_reflectance = 0.
            visualize_dict["material"] = mat

        # create wire frame as LineSet:
        wire_frame_name = None if self.name is None else self.name + "_wire_frame"
        wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
        wire_frame.paint_uniform_color(self.wire_frame_color)
        visualize_wire_frame = VisualizeLines(geometry=wire_frame,
                                              name=wire_frame_name,
                                              is_visible=self.show_wire_frame,
                                              line_width=self.wire_frame_width)

        return [visualize_dict] + visualize_wire_frame.create_visualize_dict_list()


class VisualizePoints(VisualizeObject):

    def __init__(self, point_size=8, show_normals=True, normal_width=3, normal_length=0.6,
                 normal_color="default", flip_xyz=False, **super_kwargs):
        """
        Class used for visualization of o3d.geometry.PointCloud objects.

        Parameters
        ----------
        point_size : float, optional
            Specifies point size. Note this is a global property of the viewer. The maximum is chosen.
        show_normals : bool, optional
            Specifies if normals are toggled on or off (if normals are available).
        normal_width : float, optional
            Line width of normals.
        normal_length : float, optional
            Length of normals.
        normal_color : , optional
            Color of normals.
        geometry : o3d.geometry.Geometry
            Geometry object of open3d library.
        name : str, optional
            Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.
        super_kwargs
            geometry : o3d.geometry.Geometry
            name : str, optional
                Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
            is_visible : bool, optional
                Specify if geometry should be toggled on or off in viewer. Default = True.
        """

        # call superclass
        super().__init__(**super_kwargs)
        assert isinstance(self.geometry,
                          o3d.geometry.PointCloud), "geometry must be instance of o3d.geometry.PointCloud"
        self.show_normals = show_normals
        if not self.is_visible:
            # check if show_normals was explicitly set to True:
            if "show_normals" in super_kwargs.keys():
                self.show_normals = super_kwargs["show_normals"]
            else:
                self.show_normals = False
        self.point_size = point_size
        self.normal_width = normal_width
        self.normal_length = normal_length
        self.normal_color = np.zeros(3) if normal_color == "default" else normal_color
        self.pickle_attributes = self.points_pickle_attributes
        self.flip_xyz = flip_xyz

    def flip_xyz_to_zyx(self):
        points = np.asarray(self.geometry.points)
        points = np.flip(points, axis=1)
        self.geometry.points = o3d.utility.Vector3dVector(points)

        normals = np.asarray(self.geometry.normals)
        normals = np.flip(normals, axis=1)
        self.geometry.normals = o3d.utility.Vector3dVector(normals)

    def create_visualize_dict_list(self):
        if self.flip_xyz:
            self.flip_xyz_to_zyx()

        if self.show_normals:
            if not self.geometry.has_normals():
                print("point cloud has no normals, normals are not shown.")
                self.show_normals = False

        # set points size:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = self.point_size
        mat.shader = "defaultLit"

        visualize_dict = {"geometry": self.geometry, "material": mat, "is_visible": self.is_visible}
        if self.name is not None:
            visualize_dict["name"] = self.name

        if self.geometry.has_normals():
            # create line set with normals:
            points = np.asarray(self.geometry.points)
            normals = np.asarray(self.geometry.normals)
            normal_end_points = points + self.normal_length * normals

            normal_lines = o3d.geometry.LineSet()
            normal_lines.points = o3d.utility.Vector3dVector(np.concatenate([points, normal_end_points]))
            lines = np.zeros((len(points), 2))
            lines[:, 0] = np.arange(len(points))
            lines[:, 1] = np.arange(len(points)) + len(points)
            normal_lines.lines = o3d.utility.Vector2iVector(lines)
            normal_lines.paint_uniform_color(self.normal_color)

            # create helpinstance of VisualizeLines:
            normals_name = None if self.name is None else self.name + "_normals"
            visualize_normal_lines = VisualizeLines(geometry=normal_lines,
                                                    name=normals_name,
                                                    is_visible=self.show_normals,
                                                    line_width=self.normal_width)

            return [visualize_dict] + visualize_normal_lines.create_visualize_dict_list()
        else:
            return [visualize_dict]


class VisualizeLines(VisualizeObject):

    def __init__(self, line_width=5, **super_kwargs):
        """
        Class used for visualization of o3d.geometry.PointCloud objects.

        Parameters
        ----------
        line_width : float, optional
            Width of lines in viewer. Default = 5.
        geometry : o3d.geometry.Geometry
            Geometry object of open3d library.
        name : str, optional
            Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.
        super_kwargs
            geometry : o3d.geometry.Geometry
            name : str, optional
                Name of the object as shown in the viewer. Default = f"unnamed_{unnamed_counter}".
            is_visible : bool, optional
                Specify if geometry should be toggled on or off in viewer. Default = True.
        """

        # call superclass
        super().__init__(**super_kwargs)
        assert isinstance(self.geometry, o3d.geometry.LineSet), "geometry must be instance of o3d.geometry.LineSet"

        self.line_width = line_width
        self.pickle_attributes = self.lines_pickle_attributes

    def create_visualize_dict_list(self):
        # set line width:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = self.line_width

        visualize_dict = {"geometry": self.geometry, "material": mat, "is_visible": self.is_visible}
        if self.name is not None:
            visualize_dict["name"] = self.name

        return [visualize_dict]


def end_visualization(o3dvis):
    """
    Action to end the visualization and continue the script.
    :param o3dvis: 3d.visualization.O3DVisualizer
    :return:
    """
    o3dvis.close()


def wrapped_o3d_draw(visualize_obj_list, draw_param_dict=None, path_to_tmp_file="default",
                     path_to_pyscript_folder="default", verbose=False, filter_prints=True, port_for_webvisualize=None,
                     path_to_env=None):
    """
    Wrapper to call o3d.visualization.draw as subprocess. Otherwise, I couldn't get it to work in jupyter.

    Parameters
    ----------
    visualize_obj_list : list[VisualizeObject]
        List of Objects that are visualized.
    draw_param_dict : dict, optional
        Arguments that o3d.visualization.draw takes:
        http://www.open3d.org/docs/release/python_api/open3d.visualization.draw.html.
        Default = {"width": 2560, "height": 1440, "show_skybox": False, "show_ui": False, "raw_mode": False}.
    path_to_tmp_file : str, optional
        Specifies path to temporal file which are needed to pickle inputs to communicate with subprocess.
        Default = os.path.join(PATH_TO_PYSCRIPT_FOLDER, "tmp_draw_infile.pkl").
    path_to_pyscript_folder : str, optional
        Patho to py-script that is executed as subprocess.
        Default: os.path.join(get_absolute_path_to_repo_from_inside_repo(), "cata2seg/cata2seg/").
    verbose : bool, optional
        Specifies whether all outputs of o3d.visualizer will be shown. Default : False.
    filter_prints : bool, optional
        Specifies whether logging for toggling proposals on and off is shown. Default : True.
    port_for_webvisualize : int, optional
        Specifies port to forward the visualization to. Default : None.
    path_to_env : str, optional
        Path to environment that is used to execute the subprocess. Default : None.
    Returns
    -------

    """
    name_of_pyscript = "o3d_draw.py"

    if path_to_pyscript_folder == "default":
        path_to_pyscript_folder = os.path.join(get_absolute_path_to_repo(), "deconstruct.utils.open3d_utils")
    if path_to_tmp_file == "default":
        path_to_tmp_file = os.path.join(path_to_pyscript_folder, "tmp_draw_infile.pkl")
    assert os.path.exists(os.path.join(path_to_pyscript_folder, name_of_pyscript)), \
        f"o3d_draw.py script was not found, it is expected to be located at " \
        f"{os.path.join(path_to_pyscript_folder, name_of_pyscript)}."

    print("path_to_pyscript_folder", path_to_pyscript_folder, path_to_tmp_file)

    # update standrad draw parameters
    draw_param_dict = {} if draw_param_dict is None else draw_param_dict
    standard_draw_params = {"width": 2560, "height": 1440, "show_skybox": False, "show_ui": False, "raw_mode": False}
    webrtc_standard_draw_params = {"width": 1024, "height": 768, "show_skybox": False, "show_ui": False}

    if port_for_webvisualize is not None:
        standard_draw_params.update(webrtc_standard_draw_params)

    standard_draw_params.update(draw_param_dict)
    draw_param_dict = standard_draw_params

    if not isinstance(visualize_obj_list, list):
        visualize_obj_list = [visualize_obj_list]

    # create input file and call draw as subprocess:
    input_dict = {
        "visualize_obj_list": visualize_obj_list,
        "draw_param_dict": draw_param_dict,
        "port_for_webvisualize": port_for_webvisualize,
    }

    # store in file:
    if not os.path.exists(os.path.dirname(path_to_tmp_file)):
        os.makedirs(os.path.dirname(path_to_tmp_file))
    with open(path_to_tmp_file, 'wb') as f:
        pickle.dump(input_dict, f)

    exec_list = ["python", name_of_pyscript, "-path", path_to_tmp_file]  # execute py-script
    if path_to_env is not None:
        python_interpreter = os.path.join(path_to_env, "bin/python")
        exec_list[0] = python_interpreter
    # print("exec_list", exec_list)
    process = subprocess.Popen(exec_list, cwd=path_to_pyscript_folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)

    with process.stdout:
        for line in process.stdout:
            if "[print_this]" in line and filter_prints:
                print(line.replace("[print_this]", "").replace("\n", "\r"))
            else:
                if verbose:
                    print(line.replace("\n", "\r"))


parser = argparse.ArgumentParser()
parser.add_argument('-path', help='file path to input', required=True)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    tmp_file_path = args["path"]

    # load input file:
    with open(tmp_file_path, 'rb') as f:
        input_dict = pickle.load(f)

    # delete tmp file:
    os.remove(tmp_file_path)

    # create visualize_dict_list:
    visualize_dict_list = []
    for visualize_object in input_dict["visualize_obj_list"]:
        visualize_dict_list += visualize_object.create_visualize_dict_list()

    # adding names where necessary:
    unnamed_counter = 0
    point_sizes_list = []
    for i, visualize_dict in enumerate(visualize_dict_list):
        if not "name" in visualize_dict:
            visualize_dict["name"] = f"unnamed_{unnamed_counter}"  # adding a name is necessary
            unnamed_counter += 1

        if "material" in visualize_dict:
            if hasattr(visualize_dict["material"], "point_size"):
                point_sizes_list.append(visualize_dict["material"].point_size)

    # the point size is a global property so set it to the max:
    for visualize_dict in visualize_dict_list:
        if "material" in visualize_dict:
            visualize_dict["material"].point_size = np.max(point_sizes_list)

    if input_dict["port_for_webvisualize"] is not None:
        os.environ["WEBRTC_PORT"] = str(input_dict["port_for_webvisualize"])
        o3d.visualization.webrtc_server.enable_webrtc()
        # add a button to exit the visualization:
        if "actions" in input_dict["draw_param_dict"]:
            input_dict["draw_param_dict"]["actions"].append(("Exit", end_visualization))
        else:
            input_dict["draw_param_dict"]["actions"] = [("Exit", end_visualization)]
        print(f"[print_this]WebVisualizer is using port: {input_dict['port_for_webvisualize']}", flush=True)
        print("[print_this]Press 'Exit' to end the visualization.", flush=True)
        # print("[print_this]", input_dict["draw_param_dict"], flush=True)

    o3d.visualization.draw(visualize_dict_list, **input_dict["draw_param_dict"])
