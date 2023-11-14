import sys
from maestrowf.datastructures.core import ParameterGenerator
from sklearn.model_selection import ParameterGrid
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def get_custom_generator(env, **kwargs):
      """
      Create a custom populated ParameterGenerator.
      
      This function recreates the exact same parameter set as the sample LULESH
      specifications. The point of this file is to present an example of how to
      generate custom parameters.
      
      :returns: A ParameterGenerator populated with parameters.
      """
      import sys
      p_gen = ParameterGenerator()

      yaml_file = ""
      for arg in sys.argv:
          if arg.endswith(".yaml"):
              yaml_file = arg

      yml = yaml.load(open(yaml_file).read(), Loader=Loader)
      p_in = {}
      labels = {}

      linked_ps = {}
      linked = {}

      for key, val in yml["generator.parameters"].items():

          if "values" in val:
              if isinstance(val["values"], (list,tuple)):
                  p_in[key] = list(val["values"])
              else:
                  p_in[key] = [val["values"],]

              labels[key] = val["label"]

          elif key == "LINKED":
              linked = val

      for plink in linked:
          for link in linked[plink]:
              if not plink in linked_ps:
                  linked_ps[plink] = {}
              linked_ps[plink][link] = p_in.pop(link)

      grid = ParameterGrid(p_in)
      p = {}
      for g in grid:
          for key in g:
              if key not in p:
                  p[key] = [g[key], ]
                  if key in linked_ps:
                      for link in linked_ps[key]:
                          p[link] = [linked_ps[key][link][p_in[key].index(g[key])],]
              else:
                  p[key].append(g[key])
                  if key in linked_ps:
                      for link in linked_ps[key]:
                          p[link].append(linked_ps[key][link][p_in[key].index(g[key])])


      for key, val in p.items():
          labels[key] = labels[key] + "_%%"

      for key, val in p.items():
          p_gen.add_parameter(key, val, labels[key])
      return p_gen
