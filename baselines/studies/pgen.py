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

      for k, val in yml["generator.parameters"].items():
          if "values" in val:
              if isinstance(val["values"], (list,tuple)):
                  p_in[k] = list(val["values"])
              else:
                  p_in[k] = [val["values"],]
              labels[k] = val["label"]
          elif k == "LINKED":
              linked = val

      for plink in linked:
          for link in linked[plink]:
              if not plink in linked_ps:
                  linked_ps[plink] = {}
              linked_ps[plink][link] = p_in.pop(link)

      grid = ParameterGrid(p_in)
      p = {}
      for g in grid:
          for k in g:
              if k not in p:
                  p[k] = [g[k], ]
                  if k in linked_ps:
                      for link in linked_ps[k]:
                          p[link] = [linked_ps[k][link][p_in[k].index(g[k])],]
              else:
                  p[k].append(g[k])
                  if k in linked_ps:
                      for link in linked_ps[k]:
                          p[link].append(linked_ps[k][link][p_in[k].index(g[k])])


      for k, val in p.items():
          labels[k] = labels[k] + "_%%"

      for k, val in p.items():
          p_gen.add_parameter(k, val, labels[k])
      return p_gen
