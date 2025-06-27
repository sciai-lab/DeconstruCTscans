# the code here does the following:
# - !IMPORTANT: before pasting this to the FreeCAD console select all the parts!
# - it exports a mesh-file (.stl) for each part, one for each class.
# - it exports the part info of every part (color, position), one for each instance. This is used to reconstruct the part.

# copy and paste this into the python console in FreeCAD (View, Panels, Python Console)

# imports which FreeCAD requires:
import FreeCAD, FreeCADGui, Mesh, Part

# do not import App, this seems to break stuff at least under Ubuntu
# it seems that automatically App = FreeCAD and Gui = FreeCADGui
# after replacing these the script works just fine.

# other imports:
import os
import re
import json

preselect_parts_by_name = True
use_assembly_name = True
export_name = "trice_full"
export_path = ("fill_in")
if not os.path.exists(export_path):
    os.makedirs(export_path)

# to ensure FreeCAD continues working normally after the imports
App = FreeCAD
Gui = FreeCADGui

def export(selected: list) -> None:
    """
    This function creates the CAD part catalogue and an instruction
    how to create the selected model from the individual parts.
    """
    # it is important to store part labels that have already been visited
    # as there are plenty of cross-references in the STEP file.
    processed = list()
  
    # there are some bendable parts and parts which are split into sub-parts
    # these parts cannot be identified by their part number alone
    exceptional_parts = ["(decals)", "bent"]
    is_ordinary_part = lambda x: len([f for f in exceptional_parts if f in x]) == 0
    # true if none of the exceptional strings appear!!
  
    parts = list()
    while len(selected) > 0:
        part = selected.pop()
        if isinstance(part, Part.Feature):
            # the following will be executed for individual parts.
            if preselect_parts_by_name:
                # extract the part number for the current part
                # the stickers which are also part of the STEP file do not match this pattern
                match = re.match(r"^([\w\d]+)-([a-zA-Z]+)", part.Label)
                # match tries to match the pattern at the start of a string, here: part.Label
                # \w is a special class called "word characters". It is shorthand for [a-zA-Z0-9_].
                # \d all numbers 0 to 9, I think it is unnecessary here.
                # r"" means raw
                # []+ means multiple characters of this type
                # the () brackets define the groups. There are two groups here.
                # The first group (1) is the part number, the second one (2) the color.
                # the match here will filter out the stickers which have names like: "1R".

                if match is None or part.Label in processed:
                    if part.Label in processed:
                        print(f"part {part.Label} already processed. When would this happen?")
                    # go to next part in the selection
                    continue
                label = match.group(1) if is_ordinary_part(part.Label) else part.Label
            else:
                match = None
                label = part.Label
            processed.append(part.Label)
            # part number is used as label for ordinary parts, otherwise use the full name.
            print("part appended ", label)
            parts.append(
              {
                "part": label,
                "material": "unknown" if match is None else match.group(2),  # this is actually the color
                "global_offset": list(FreeCAD.ActiveDocument.RootObjects[0].Shape.BoundBox.Center),  # global origin (3,)
                "transformation": list(part.getGlobalPlacement().toMatrix().A)  # shape (16,)
              }
            )
      
            part_path = os.path.join(export_path, f"{label}.stl")
            if not os.path.exists(part_path):
                # to export the part for the CAD catalogue remove any transformations
                tmp = part.Placement
                part.Placement = FreeCAD.Placement()  # set the origin of the part to (0,0,0) for the export
                print("exported to ", part_path)
                Mesh.export([part], part_path)  # exports the part
                part.Placement = tmp
      
        elif isinstance(part, FreeCAD.Part):
            # the following will be executed for assemblies of parts, e.g. "Assy01".
            
            processed.append(part.Label)
            for s in part.getSubObjects():
                for i, p in enumerate(part.getSubObjectList(s)):
                    # now we are again at the level of individual parts.
                    # print(len(part.getSubObjectList(s)))
                    # print(list(part.getSubObjectList(s)))
                    # print([p.Label for p in part.getSubObjectList(s)])
                    if use_assembly_name and len(part.getSubObjectList(s)) == 2:
                        if isinstance(part.getSubObjectList(s)[1], Part.Feature):
                            # if the assembly has only one part, use the assembly name as part name
                            print(f"changing label from {p.Label} to {part.Label}")
                            p.Label = part.Label

                    if p.Label not in processed:
                        selected.append(p)
                    else:
                        print("part already processed ", p.Label)
  
    with open(os.path.join(export_path, f"{export_name}_info.json"), "w") as f:
        # export all relevant part information (including the transformations to reassemble them):
        json.dump(parts, f)
        

selected = FreeCADGui.Selection.getSelection()
export(selected)