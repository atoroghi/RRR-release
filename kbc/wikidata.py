import requests
import copy, sys

class Wikidata:
    def __init__(self):
        pass

    def get_triples(self, entity, is_label):
        if is_label:
            qid = self._label_to_id(entity)
        else:
            qid = entity
        
        return self._get_triples_from_qid(qid)

    def get_description(self, entity, is_label):
        if is_label:
            qid = self._label_to_id(entity)
        else:
            qid = entity 
        
        response = self._get_entity(qid)

        if not response:
            return

        try:
            entity_description = response["entities"][qid]["descriptions"]["en"]["value"]
            return entity_description
        except:
            return  
        


    def _get_triples_from_qid(self, qid):
        response = self._get_entity(qid)

        if not response:
            return

        triples = []
        tail_label_to_id = {} 

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
            if len(triples) > 100:
                break
            property_label = self._id_to_label(property_id)

            for statement in statements:
                


                mainsnak = statement["mainsnak"]
                if mainsnak["snaktype"] != "value":
                    continue
                
                if mainsnak["datatype"] == "wikibase-item":
                    tail_id, tail_label = self._get_snak_value(mainsnak)
                    mainsnak_value = tail_label
                else:
                    mainsnak_value = self._get_snak_value(mainsnak)
                
                if not mainsnak_value:
                    continue
                
                if mainsnak["datatype"] == "wikibase-item":
                    tail_label_to_id[tail_label] = tail_id

                triple = [[entity_label, property_label, mainsnak_value]]

                # qualifiers
                if not "qualifiers" in statement:
                    triples.append(triple)
                    continue

                qualifiers = statement["qualifiers"]

                for qualifier_property_id, qualifier_snaks in qualifiers.items():
                    qualifier_property_label = self._id_to_label(qualifier_property_id)

                    for qualifier_snak in qualifier_snaks:
                        if qualifier_snak["snaktype"] != "value":
                            continue
                        if qualifier_snak["datatype"] == "wikibase-item":
                            qualifier_tail_id, qualifier_tail_label = self._get_snak_value(qualifier_snak)
                            qualifier_snak_value = qualifier_tail_label
                        else:
                            qualifier_snak_value = self._get_snak_value(qualifier_snak)
                            
                        if not qualifier_snak_value:
                            continue

                        if qualifier_snak["datatype"] == "wikibase-item":
                            tail_label_to_id[qualifier_tail_label] = qualifier_tail_id

                        qualifier = [qualifier_property_label, qualifier_snak_value]
                        triple.append(qualifier)

                triples.append(triple)

        # verbalized_triples = self.verbalize_triples(triples)
        # return verbalized_triples
                
        return triples, tail_label_to_id

    def verbalize_triples(self, triples):
        """
        [[head, relation, tail], [qualifier_relation, qualifier_tail], ...]
        "head, relation, tail; qualifier_relation, qualifier_tail; ..."
        """
        str_triples = copy.deepcopy(triples)

        for i in range(len(str_triples)):
            str_triples[i] = self._verbalize_triple(str_triples[i])

        return str_triples
    
    def _verbalize_triple(self, triple):
        for j in range(len(triple)):
            triple[j] = ", ".join(triple[j])

        triple = "; ".join(triple)
        triple = f"({triple})"

        return triple 


    
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
    
    def _label_to_id(self, label):
        response = self._search_label(label)
        if not response:
            return

        search_results = response["search"]
        if not search_results:
            return 

        qid = search_results[0]["id"]  # TODO: assumes first

        return qid


    def _get_snak_value(self, snak):
        try:
            datatype = snak["datatype"]
            datavalue = snak["datavalue"]["value"]
        except:
            return None

        match datatype:
            case "wikibase-item":
                tail_id = datavalue["id"]
                tail_label = self._id_to_label(tail_id)
                snak_value = (tail_id, tail_label)

            case "time":

                if datavalue["time"][6] + datavalue["time"][7] == "00":
                    snak_value = datavalue["time"][1:5]
                else:
                    snak_value = datavalue["time"][1:5] + "/" + datavalue["time"][6:8] + "/" + datavalue["time"][9:11]

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

    