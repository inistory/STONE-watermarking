# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated 
#              as one of the watermark classes of the library when created 
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import importlib

WATERMARK_MAPPING_NAMES={
    'STONE': 'watermark.stone.STONE',
    'KGW': 'watermark.kgw.KGW',
    'SWEET': 'watermark.sweet.SWEET',
    'EWD': 'watermark.ewd.EWD',
}

def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name == algorithm_name:
            return watermark_name
    return None

class STONEAutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(algorithm_name, transformers_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)

        skipping_rule = kwargs.get('skipping_rule')
        watermark_on_pl = kwargs.get('watermark_on_pl')
        gamma = kwargs.get('gamma')
        delta = kwargs.get('delta')
        hash_key = kwargs.get('hash_key')
        prefix_length = kwargs.get('prefix_length')
        z_threshold = kwargs.get('z_threshold')
        language = kwargs.get('language')

        watermark_instance = watermark_class(transformers_config, skipping_rule=skipping_rule, watermark_on_pl=watermark_on_pl, gamma=gamma, delta=delta, hash_key=hash_key, prefix_length=prefix_length, z_threshold=z_threshold, language=language)
        return watermark_instance
    
class AutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(algorithm_name, transformers_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)

        gamma = kwargs.get('gamma')
        delta = kwargs.get('delta')
        hash_key = kwargs.get('hash_key')
        prefix_length = kwargs.get('prefix_length')
        z_threshold = kwargs.get('z_threshold')
        f_scheme = kwargs.get('f_scheme')
        window_scheme = kwargs.get('window_scheme')
        entropy_threshold = kwargs.get('entropy_threshold')

        watermark_instance = watermark_class(transformers_config, gamma=gamma, delta=delta, 
                                             hash_key=hash_key, prefix_length=prefix_length, entropy_threshold=entropy_threshold, 
                                             z_threshold=z_threshold, f_scheme=f_scheme, window_scheme=window_scheme)
        
        return watermark_instance