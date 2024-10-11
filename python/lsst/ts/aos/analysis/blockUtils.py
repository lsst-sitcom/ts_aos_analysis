# This file is part of ts_aos_analysis.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ['add_property', 'build_configuration_schema']

def add_property(prop_name: str, prop_details: dict, indent_level: int = 1) -> str:
    """Add a property to a configuration json-formatted schema string.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    prop_details : dict
        A dictionary with keys 'description', 'type', and optionally 'default'.
    indent_level : int, optional
        The number of indentation levels to add to the property, by default 1.

    Returns
    -------
    str
        The formatted property string.
    """
    indent = '  ' * indent_level
    schema_str = f'{indent}{prop_name}:\n'
    schema_str += f'{indent}  description: {prop_details["description"]}\n'
    schema_str += f'{indent}  type: {prop_details["type"]}\n'

    # Handle default values
    if "default" in prop_details:
        default_value = prop_details["default"]
        if prop_details["type"] == "string":
            default_value = f'"{default_value}"'
        schema_str += f'{indent}  default: {default_value}\n'

    # Handle nested properties for arrays or objects
    if prop_details["type"] == "array" and "items" in prop_details:
        schema_str += f'{indent}  items:\n'
        # Add the array item type without the description
        item_details = prop_details["items"]
        schema_str += f'{indent}    type: {item_details["type"]}\n'
        if "minimum" in item_details:
            schema_str += f'{indent}    minimum: {item_details["minimum"]}\n'
        if "maximum" in item_details:
            schema_str += f'{indent}    maximum: {item_details["maximum"]}\n'

    if prop_details["type"] == "object" and "properties" in prop_details:
        schema_str += f'{indent}  properties:\n'
        for nested_name, nested_details in prop_details["properties"].items():
            schema_str += add_property(nested_name, nested_details, indent_level + 2)

    return schema_str

def build_configuration_schema(block_number: int, properties: dict) -> str:
    """
    Builds a configuration schema string for a given BLOCK and configurable properties.

    Parameters
    ----------
    block_number : int
        The BLOCK number for which the configuration schema is being built.
    properties : dict
        A dictionary with property names as keys and dictionaries as values.
        The dictionaries should have keys 'description', 'type', and optionally 'default'.

    Returns
    -------
    str
        A json-formatted configuration schema string.
    """
    # Define the base schema with the BLOCK number
    configuration_schema = (
        '$schema: http://json-schema.org/draft-07/schema#\n'
        f'title: BLOCK-{block_number} configuration\n'
        f'description: Configuration for BLOCK-{block_number}.\n'
        'type: object\n'
        'properties:\n'
    )

    # Add each property to the schema
    for prop_name, prop_details in properties.items():
        configuration_schema += add_property(prop_name, prop_details)

    return configuration_schema
