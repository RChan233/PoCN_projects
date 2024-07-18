import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import pandas as pd
import geopandas
from pathlib import Path
import warnings
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# Dictionary for country abbreviations
country_dict = {"AL": ("Albania", "ALB"), "AT": ("Austria", "AUT"), "BE": ("Belgium", "BEL"), "BG": ("Bulgaria", "BGR"), "CHLI": ("Switzerland", "CHE"), "CY": ("Cyprus", "CYP"),
                    "CZ": ("Czechia", "CZE"), "DE": ("Germany", "DEU"), "DK": ("Denmark", "DNK"), "EE": ("Estonia", "EST"), "ES": ("Spain", "ESP"), "FI": ("Finland", "FIN"),
                    "FR": ("France", "FRA"), "GB": ("United Kingdom of Great Britain and Northern Ireland", "GBR"), "GE": ("Georgia", "GEO"), "GR": ("Greece", "GRC"), 
                    "HR": ("Croatia", "HRV"), "HU": ("Hungary", "HUN"), "IE": ("Ireland", "IRL"), "IS": ("Iceland", "IS"), "LT": ("Lithuania", "LTU"), "SK": ("Slovakia", "SVK"),
                    "PT": ("Portugal", "PRT"), "RS": ("Serbia", "SRB"), "PL": ("Poland", "POL"), "IT": ("Italy", "ITA"), "LU": ("Luxembourg", "LUX"), "CH": ("Switzerland", "CHE"),
                    "NL": ("Netherlands", "NLD"), "RO": ("Romania", "ROU"), "ND": ("NA", "NA"), "MD": ("Moldova", "MDA"), "NO": ("Norway", "NOR"), "SE": ("Sweden", "SWE"),
                    "LV": ("Latvia", "LVA"), "SI": ("Slovenia", "SVN"), "UA": ("Ukraine", "UKR"), "MK": ("Republic of North Macedonia", "MKD"), "LI": ("Liechtenstein", "LIE"), 
                    "FO": ("Faroe Islands", "FRO"), "MT": ("Malta", "MLT")}


def buildEdgesGeoDF(country_abr, europe=False):
    filelocStr = "DATA/Countries/"
    
    edges_info = geopandas.GeoDataFrame()
    edges_loc = filelocStr + country_abr + "/RailrdL.shp"

    if europe:
        edges_loc = "DATA/FullEurope/RailrdL.shp"
    edges_info = geopandas.read_file(edges_loc)

    return edges_info
    
def buildNodesGeoDF(country_abr, europe=False):
    filetypes = [("RailrdC", "train_station"), ("ExitC", "entrance_or_exit"), ("FerryC", "ferry_station"), ("LevelcC", "level_crossing"), ("AirfldP", "airfield")]
    filelocStr = "DATA/Countries/"
    
    nodes_info = geopandas.GeoDataFrame()
    for type, label in filetypes:
        if not europe:
            nodes_loc = filelocStr + country_abr + "/" + type + ".shp"
        else:
            nodes_loc = "DATA/FullEurope/" + type + ".shp"

        if Path(nodes_loc).is_file():
            temp_nodes_df = geopandas.read_file(nodes_loc)
            temp_nodes_df["nodeLabel"] = label
            nodes_info = pd.concat([nodes_info, temp_nodes_df], ignore_index=True)
    
    return nodes_info

def fillNodesDF(edge_crds, country_name, country_ISO3):
    nodes_df = geopandas.GeoDataFrame(columns=["nodeID", "nodeLabel", "latitude", "longitude", "country_name", "country_ISO3"])
    nodes_df["latitude"] = edge_crds["geometry"].x
    nodes_df["longitude"] = edge_crds["geometry"].y
    nodes_df["nodeID"] = nodes_df.index
    nodes_df["nodeLabel"] = "unknown"
    nodes_df["country_name"] = country_name
    nodes_df["country_ISO3"] = country_ISO3

    nodes_df["geometry"] = edge_crds["geometry"]

    return nodes_df

def fillEdgeDF(nodes_df, edges_info):
    edges_df = pd.DataFrame(columns=["NodeID_from", "NodeID_to", "weight"])

    edges_boundaries = edges_info["geometry"].boundary
    exploded = edges_boundaries.explode()
    start_crds = exploded[:,0]
    end_crds = exploded[:,1]

    for s_crd, e_crd in zip(start_crds, end_crds):
        startID = np.where(nodes_df["geometry"] == s_crd)[0][0]
        endID = np.where(nodes_df["geometry"] == e_crd)[0][0]
        new_row = pd.DataFrame([[startID, endID, 1]], columns=["NodeID_from", "NodeID_to", "weight"])
        edges_df = pd.concat([edges_df, new_row], ignore_index=True)

    edges_df["NodeID_from"] = edges_df["NodeID_from"].astype(int)
    edges_df["NodeID_to"] = edges_df["NodeID_to"].astype(int)
    edges_df["weight"] = 1
    
    return edges_df
    
def getEdgeCrds(edges_info):
    edges_boundaries = edges_info["geometry"].boundary
    exploded = edges_boundaries.explode()
    crds1 = geopandas.GeoDataFrame()
    crds2 = geopandas.GeoDataFrame()
    crds1["geometry"] = exploded[:,0].values
    crds2["geometry"] = exploded[:,1].values
    crds = pd.concat([crds1, crds2], ignore_index=True)
    crds = crds.drop_duplicates(ignore_index=True)

    return crds


def cleanData(country_abr, countryDict_vals):
    if not Path(f"DATA/Countries/{country_abr}/RailrdL.shp").is_file():
        print("Country not in list")
        return

    # Check if file already exists
    path_nodes = Path(f'Cleaned_data/Countries/{country_abr}/nodes.csv')
    path_edges = Path(f'Cleaned_data/Countries/{country_abr}/edges.csv')
    if path_nodes.is_file():
        print("Cleaned file already exists")
        return

    country_name, country_ISO3 = countryDict_vals

    edges_info = buildEdgesGeoDF(country_abr)
    nodes_info = buildNodesGeoDF(country_abr)

    edge_crds = getEdgeCrds(edges_info)

    nodes_df = fillNodesDF(edge_crds, country_name, country_ISO3)
    
    # Add labels to known nodes
    for i, crd in enumerate(nodes_df["geometry"]):
        info_index = nodes_info.loc[nodes_info["geometry"] == crd]["geometry"].index.tolist()
        if info_index:
            nodes_df.loc[i, "nodeLabel"] = nodes_info.loc[info_index[0], "nodeLabel"]


    edges_df = fillEdgeDF(nodes_df, edges_info)
    nodes_df = pd.DataFrame(nodes_df.drop(columns="geometry"))

    # Save cleaned files
    path_nodes.parent.mkdir(parents=True, exist_ok=True)  
    path_edges.parent.mkdir(parents=True, exist_ok=True)

    nodes_df.to_csv(path_nodes, index=False)
    edges_df.to_csv(path_edges, index=False)


def cleanEuropeData():
    edges_info = buildEdgesGeoDF(_, europe=True)
    nodes_info = buildNodesGeoDF(_, europe=True)

    edge_crds = getEdgeCrds(edges_info)

    nodes_df = geopandas.GeoDataFrame(columns=["nodeID", "nodeLabel", "latitude", "longitude", "country_name", "country_ISO3"])
    nodes_df["latitude"] = edge_crds["geometry"].x
    nodes_df["longitude"] = edge_crds["geometry"].y
    nodes_df["nodeID"] = nodes_df.index
    nodes_df["nodeLabel"] = "unknown"
    nodes_df["country_name"] = "NA"
    nodes_df["country_ISO3"] = "NA"
    nodes_df["geometry"] = edge_crds["geometry"]
    
    # Add labels and countries to known nodes
    for i, crd in enumerate(nodes_df["geometry"]):
        info_index = nodes_info.loc[nodes_info["geometry"] == crd]["geometry"].index.tolist()
        if info_index:
            nodes_df.loc[i, "nodeLabel"] = nodes_info.loc[info_index[0], "nodeLabel"]
            if nodes_info.loc[info_index[0], "ICC"] in country_dict.keys():
                name, code_ISO3 = country_dict[nodes_info.loc[info_index[0], "ICC"]]
                nodes_df.loc[i, "country_name"] = name
                nodes_df.loc[i, "country_ISO3"] = code_ISO3

    edges_df = fillEdgeDF(nodes_df, edges_info)
    nodes_df = pd.DataFrame(nodes_df.drop(columns="geometry"))

    # Save cleaned files
    path_nodes = Path(f'Cleaned_data/FullEurope/nodes.csv')
    path_edges = Path(f'Cleaned_data/FullEurope/edges.csv')
    
    path_nodes.parent.mkdir(parents=True, exist_ok=True)  
    path_edges.parent.mkdir(parents=True, exist_ok=True)  

    nodes_df.to_csv(path_nodes, index=False)
    edges_df.to_csv(path_edges, index=False)


# Clean country data
for key, vals in country_dict.items():
    cleanData(key, vals)

# Clean Europe data
cleanEuropeData()


