""" Handy utility functions."""

from __future__ import annotations

import asyncio
import inspect
import json
import json.decoder
import os
import random
import sys
import warnings
from copy import deepcopy
from distutils.version import StrictVersion
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type

import aiohttp
import analytics
import httpx
import pkg_resources
import requests
from httpx import AsyncClient, Response
from pydantic import BaseModel, parse_obj_as, Json

import gradio

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from gradio import Blocks, Interface
    from gradio.blocks import BlockContext
    from gradio.components import Component

analytics_url = "https://api.gradio.app/"
PKG_VERSION_URL = "https://api.gradio.app/pkg-version"
analytics.write_key = "uxIFddIEuuUcFLf9VgH2teTEtPlWdkNy"
JSON_PATH = os.path.join(os.path.dirname(gradio.__file__), "launches.json")


def version_check():
    try:
        current_pkg_version = pkg_resources.require("gradio")[0].version
        latest_pkg_version = requests.get(url=PKG_VERSION_URL).json()["version"]
        if StrictVersion(latest_pkg_version) > StrictVersion(current_pkg_version):
            print(
                "IMPORTANT: You are using gradio version {}, "
                "however version {} "
                "is available, please upgrade.".format(
                    current_pkg_version, latest_pkg_version
                )
            )
            print("--------")
    except pkg_resources.DistributionNotFound:
        warnings.warn(
            "gradio is not setup or installed properly. Unable to get version info."
        )
    except json.decoder.JSONDecodeError:
        warnings.warn("unable to parse version details from package URL.")
    except KeyError:
        warnings.warn("package URL does not contain version info.")
    except:
        pass


def get_local_ip_address() -> str:
    try:
        ip_address = requests.get("https://api.ipify.org", timeout=3).text
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        ip_address = "No internet connection"
    return ip_address


def initiated_analytics(data: Dict[str:Any]) -> None:
    try:
        requests.post(
            analytics_url + "gradio-initiated-analytics/", data=data, timeout=3
        )
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        pass  # do not push analytics if no network


def launch_analytics(data: Dict[str, Any]) -> None:
    try:
        requests.post(
            analytics_url + "gradio-launched-analytics/", data=data, timeout=3
        )
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        pass  # do not push analytics if no network


def integration_analytics(data: Dict[str, Any]) -> None:
    try:
        requests.post(
            analytics_url + "gradio-integration-analytics/", data=data, timeout=3
        )
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        pass  # do not push analytics if no network


def error_analytics(ip_address: str, message: str) -> None:
    """
    Send error analytics if there is network
    :param type: RuntimeError or NameError
    """
    data = {"ip_address": ip_address, "error": message}
    try:
        requests.post(analytics_url + "gradio-error-analytics/", data=data, timeout=3)
    except (requests.ConnectionError, requests.exceptions.ReadTimeout):
        pass  # do not push analytics if no network


async def log_feature_analytics(ip_address: str, feature: str) -> None:
    data = {"ip_address": ip_address, "feature": feature}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                analytics_url + "gradio-feature-analytics/", data=data
            ):
                pass
        except (aiohttp.ClientError):
            pass  # do not push analytics if no network


def colab_check() -> bool:
    """
    Check if interface is launching from Google Colab
    :return is_colab (bool): True or False
    """
    is_colab = False
    try:  # Check if running interactively using ipython.
        from IPython import get_ipython

        from_ipynb = get_ipython()
        if "google.colab" in str(from_ipynb):
            is_colab = True
    except (ImportError, NameError):
        pass
    return is_colab


def ipython_check() -> bool:
    """
    Check if interface is launching from iPython (not colab)
    :return is_ipython (bool): True or False
    """
    is_ipython = False
    try:  # Check if running interactively using ipython.
        from IPython import get_ipython

        if get_ipython() is not None:
            is_ipython = True
    except (ImportError, NameError):
        pass
    return is_ipython


def readme_to_html(article: str) -> str:
    try:
        response = requests.get(article, timeout=3)
        if response.status_code == requests.codes.ok:  # pylint: disable=no-member
            article = response.text
    except requests.exceptions.RequestException:
        pass
    return article


def show_tip(interface: gradio.Blocks) -> None:
    if interface.show_tips and random.random() < 1.5:
        tip: str = random.choice(gradio.strings.en["TIPS"])
        print(f"Tip: {tip}")


def launch_counter() -> None:
    try:
        if not os.path.exists(JSON_PATH):
            launches = {"launches": 1}
            with open(JSON_PATH, "w+") as j:
                json.dump(launches, j)
        else:
            with open(JSON_PATH) as j:
                launches = json.load(j)
            launches["launches"] += 1
            if launches["launches"] in [25, 50, 150, 500, 1000]:
                print(gradio.strings.en["BETA_INVITE"])
            with open(JSON_PATH, "w") as j:
                j.write(json.dumps(launches))
    except:
        pass


def get_default_args(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    return [
        v.default if v.default is not inspect.Parameter.empty else None
        for v in signature.parameters.values()
    ]


def assert_configs_are_equivalent_besides_ids(config1, config2):
    """Allows you to test if two different Blocks configs produce the same demo."""
    assert config1["mode"] == config2["mode"], "Modes are different"
    assert config1["theme"] == config2["theme"], "Themes are different"
    assert len(config1["components"]) == len(
        config2["components"]
    ), "# of components are different"

    mapping = {}

    for c1, c2 in zip(config1["components"], config2["components"]):
        c1, c2 = deepcopy(c1), deepcopy(c2)
        mapping[c1["id"]] = c2["id"]
        c1.pop("id")
        c2.pop("id")
        assert c1 == c2, "{} does not match {}".format(c1, c2)

    def same_children_recursive(children1, chidren2, mapping):
        for child1, child2 in zip(children1, chidren2):
            assert mapping[child1["id"]] == child2["id"], "{} does not match {}".format(
                child1, child2
            )
            if "children" in child1 or "children" in child2:
                same_children_recursive(child1["children"], child2["children"], mapping)

    children1 = config1["layout"]["children"]
    children2 = config2["layout"]["children"]
    same_children_recursive(children1, children2, mapping)

    for d1, d2 in zip(config1["dependencies"], config2["dependencies"]):
        for t1, t2 in zip(d1["targets"], d2["targets"]):
            assert mapping[t1] == t2, "{} does not match {}".format(d1, d2)
        assert d1["trigger"] == d2["trigger"], "{} does not match {}".format(d1, d2)
        for i1, i2 in zip(d1["inputs"], d2["inputs"]):
            assert mapping[i1] == i2, "{} does not match {}".format(d1, d2)
        for o1, o2 in zip(d1["outputs"], d2["outputs"]):
            assert mapping[o1] == o2, "{} does not match {}".format(d1, d2)

    return True


def format_ner_list(input_string: str, ner_groups: Dict[str : str | int]):
    if len(ner_groups) == 0:
        return [(input_string, None)]

    output = []
    prev_end = 0

    for group in ner_groups:
        entity, start, end = group["entity_group"], group["start"], group["end"]
        output.append((input_string[prev_end:start], None))
        output.append((input_string[start:end], entity))
        prev_end = end

    output.append((input_string[end:], None))
    return output


def delete_none(_dict):
    """
    Delete None values recursively from all of the dictionaries, tuples, lists, sets.
    Credit: https://stackoverflow.com/a/66127889/5209347
    """
    if isinstance(_dict, dict):
        for key, value in list(_dict.items()):
            if isinstance(value, (list, dict, tuple, set)):
                _dict[key] = delete_none(value)
            elif value is None or key is None:
                del _dict[key]

    elif isinstance(_dict, (list, set, tuple)):
        _dict = type(_dict)(delete_none(item) for item in _dict if item is not None)

    return _dict


def resolve_singleton(_list):
    if len(_list) == 1:
        return _list[0]
    else:
        return _list


def component_or_layout_class(cls_name: str) -> Component | BlockContext:
    """
    Returns the component, template, or layout class with the given class name, or
    raises a ValueError if not found.

    Parameters:
    cls_name (str): lower-case string class name of a component
    Returns:
    cls: the component class
    """
    import gradio.components
    import gradio.layouts
    import gradio.templates

    components = [
        (name, cls)
        for name, cls in gradio.components.__dict__.items()
        if isinstance(cls, type)
    ]
    templates = [
        (name, cls)
        for name, cls in gradio.templates.__dict__.items()
        if isinstance(cls, type)
    ]
    layouts = [
        (name, cls)
        for name, cls in gradio.layouts.__dict__.items()
        if isinstance(cls, type)
    ]
    for name, cls in components + templates + layouts:
        if name.lower() == cls_name.replace("_", "") and (
            issubclass(cls, gradio.components.Component)
            or issubclass(cls, gradio.blocks.BlockContext)
        ):
            return cls
    raise ValueError(f"No such component or layout: {cls_name}")


def synchronize_async(func: Callable, *args: object, callback_func: Callable = None):
    event_loop = asyncio.get_event_loop()
    task = event_loop.create_task(func(*args))
    task.add_done_callback(callback_func)


class Http:
    """
    A class that shares an AsyncClient which sends and handles network requests.
    # TODO : This class will manage Http client instance. We should consider to add functionalities:
        - a start function to start async client
        - a close function to close async client
        - restart function to restart the client
    """
    _client: AsyncClient

    def __init__(self):
        self._client = AsyncClient()

    @property
    def client(self):
        return self._client

    async def __aexit__(self, *excinfo):
        await self._client.aclose()


class Request:
    """ Request class
        - Manages requests without a context manager
        - Handles exceptions sliently during the client call
        - Employs httpx to perform client calls
        - Includes request method types for ease of usage
        - Includes validation functionality by models and functions.
    """

    class Method(str, Enum):
        """An enum class that contains the possible types of Request method strings."""
        ANY: str = "*"
        CONNECT: str = "CONNECT"
        HEAD: str = "HEAD"
        GET: str = "GET"
        DELETE: str = "DELETE"
        OPTIONS: str = "OPTIONS"
        PATCH: str = "PATCH"
        POST: str = "POST"
        PUT: str = "PUT"
        TRACE: str = "TRACE"

    def __init__(self,
                 method: Method,
                 url: str,
                 *,
                 validation_model: Type[BaseModel] = None,
                 validation_function: Callable = None,
                 exception_class: Type[Exception] = Exception,
                 raise_for_status: bool = False,
                 **kwargs):
        # Init class attributes.
        self._response = None
        self._exception = None
        self._status = None
        self._raise_for_status = raise_for_status
        self._validation_model = validation_model
        self._validation_function = validation_function
        self._exception_class = exception_class
        # Create request
        self.request = self._create_request(method, url, **kwargs)
        self.raise_for_status = raise_for_status
        self.validation_model = validation_model
        self.validation_function = validation_function
        self.exception_class = exception_class

    def __await__(self):
        """Define request's awaited behaviour"""
        return self.__run().__await__()

    async def __run(self) -> Request:
        """ A function to make request, validate and serialize
        """
        try:
            # Make request call
            # TODO : We should move client creation to the outer scope (Block's init). And we must close it properly.
            async with Http().client as client:
                response: Response = await client.send(self.request)
            # Raise for _status
            self._status = response.status_code
            if self._raise_for_status:
                response.raise_for_status()
            # Parse client response to JSON
            self._json = response.json()
            # Validate data
            self._validated_data = self._validate(self._json)
        except Exception as exception:
            # If there is an exception, store it in an attribute
            self._exception = self._exception_class(exception)
        return self

    @staticmethod
    def _create_request(method: Method,
                        url: str,
                        **kwargs):
        """Create a request. This is a httpx request wrapper function.
        Args:
            method(Request.Method): request method type
            url(str): target url of the request
            **kwargs
        """
        request = httpx.Request(method, url, **kwargs)
        return request

    def _validate(self, response):
        if self._validation_model:
            validated_data = parse_obj_as(self._validation_model, response)
        elif self._validation_function:
            validated_data = self._validation_function(response)
        else:
            # If there is no validation method, use response
            validated_data = response
        return validated_data

    def is_valid(self, raise_exceptions: bool = False):
        if not self.has_exception:
            return True
        else:
            if raise_exceptions:
                raise self._exception
            else:
                return False

    @property
    def json(self):
        return self._json

    @property
    def validated_data(self):
        return self._validated_data

    @property
    def exception(self):
        return self._exception

    @property
    def has_exception(self):
        return self.exception is not None

    @property
    def raise_exceptions(self):
        if self.has_exception:
            raise self._exception

    @property
    def status(self):
        return self._status
