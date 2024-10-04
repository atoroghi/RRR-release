import requests
import copy


class Wikidata:
    def __init__(self):
        pass

    def from_label(self, label):
        response = self._search_label(label)
        if not response:
            return

        search_results = response["search"]
        qid = search_results[0]["id"]  # TODO: assumes first

        return self.from_id(qid)

    def from_id(self, qid):
        response = self._get_entity(qid)

        if not response:
            return

        triples = []

        entity_info = response["entities"][qid]

        try:
            entity_label = entity_info["labels"]["en"]["value"]
        except:
            return  

        try:
            entity_description = entity_info["descriptions"]["en"]["value"]
            triples.append([[entity_label, "is", entity_description]])
        except:
            pass 

        entity_claims = entity_info["claims"]

        for property_id, statements in entity_claims.items():
            property_label = self._id_to_label(property_id)

            for statement in statements:
                mainsnak = statement["mainsnak"]
                if mainsnak["snaktype"] != "value":
                    continue

                mainsnak_value = self._get_snak_value(mainsnak)
                if not mainsnak_value:
                    continue

                triple = [[entity_label, property_label, mainsnak_value]]

                # qualifiers
                if not "qualifiers" in statement:
                    triples.append(triple)
                    continue

                qualifiers = statement["qualifiers"]

                for qualifier_property_id, qualifier_snaks in qualifiers.items():
                    qualifier_property_label = self._id_to_label(qualifier_property_id)

                    for qualifier_snak in qualifier_snaks:
                        qualifier_snak_value = self._get_snak_value(qualifier_snak)
                        if not qualifier_snak_value:
                            continue

                        qualifier = [qualifier_property_label, qualifier_snak_value]
                        triple.append(qualifier)

                triples.append(triple)

        return triples

    def verbalize_triples(self, triples):
        """
        [[head, relation, tail], [qualifier_relation, qualifier_tail], ...]
        "head, relation, tail; qualifier_relation, qualifier_tail; ..."
        """
        verbalized_triples = copy.deepcopy(triples)

        for i in range(len(verbalized_triples)):
            triple = verbalized_triples[i]

            for j in range(len(triple)):
                triple[j] = ", ".join(triple[j])

            triple = "; ".join(triple)
            verbalized_triples[i] = f"({triple})"

        return verbalized_triples

    
    def _valid_response(self, response):
        return response.status_code == 200

    def _search_label(self, label):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label,
        }

        response = requests.get(url, params=params)
        if self._valid_response(response):
            return response.json()

    def _get_entity(self, id):
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{id}.json"

        response = requests.get(url)
        if self._valid_response(response):
            return response.json()

    def _id_to_label(self, id):
        response = self._get_entity(id)
        if response:
            try:
                return response["entities"][id]["labels"]["en"]["value"]
            except:
                return None

    def _get_snak_value(self, snak):
        try:
            datatype = snak["datatype"]
            datavalue = snak["datavalue"]["value"]
        except:
            return None

        match datatype:
            case "wikibase-item":
                tail_id = datavalue["id"]
                snak_value = self._id_to_label(tail_id)

            case "time":
                snak_value = datavalue["time"]

            case "string":
                snak_value = datavalue

            case "quantity":
                amount = datavalue["amount"]
                unit = datavalue["unit"]

                if unit == "1":
                    snak_value = amount
                elif "wikidata.org/entity" in unit:
                    unit_id = unit.split("/")[-1]
                    unit_label = self._id_to_label(unit_id)
                    snak_value = amount + " " + unit_label

            case "globe-coordinate":
                latitude = datavalue["latitude"]
                longitude = datavalue["longitude"]
                snak_value = f"{latitude}, {longitude}"

            case "math":
                snak_value = datavalue

            case _:
                snak_value = None

        return snak_value

    