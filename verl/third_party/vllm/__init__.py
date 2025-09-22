# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib.metadata import version, PackageNotFoundError
import warnings


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)

SUPPORTED_VERSION_MAP = {
    '0.3.1': '0.3.1',
    '0.4.2': '0.4.2',
    '0.5.1': '0.5.4',  # 0.5.1 is API-compatible with the 0.5.4 shim we vendor
    '0.5.4': '0.5.4',
    '0.6.3': '0.6.3',
}

canonical_version = SUPPORTED_VERSION_MAP.get(package_version)

if canonical_version is None:
    supported_versions = ', '.join(sorted(SUPPORTED_VERSION_MAP.keys()))
    raise ValueError(
        f'vllm version {package_version} not supported. Currently supported versions are {supported_versions}.'
    )

if canonical_version != package_version:
    warnings.warn(
        f"vllm version {package_version} detected; using compatibility shim for {canonical_version}.",
        RuntimeWarning,
        stacklevel=2,
    )

if canonical_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif canonical_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif canonical_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif canonical_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
