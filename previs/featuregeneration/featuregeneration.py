import featuretools as ft


def engineer_features_using_feature_tools(data_frame, primary_key, key_variable_map):
    """This method is used to engineer and generate new features
    from an existing dataframe
    @param data_frame: The dataframe which has all the features.
    Keep note that the target feature should not be a part of this.
    Also, the dataframe should be clean and all the preprocessing should have already been done
    on it
    @param primary_key: This is the primary key of the dataframe. Usually some form of
    id.
    @param key_variable_map: This is a dictionary. Please refer to the Readme.md in featuregeneration
    @return: Returns the new dataframe constituting of new features alongside other ones as well.
    """
    entity_set = ft.EntitySet(id='entity_set_id')
    entity_set.entity_from_dataframe(entity_id='entity_id_1', dataframe=data_frame, index=primary_key)

    counter = 1
    previous_entity_name = 'entity_id_' + str(counter)
    for key in key_variable_map:
        variable_list = key_variable_map[key]
        counter += 1
        new_entity_name = 'entity_id_' + str(counter)
        entity_set.normalize_entity(base_entity_id=previous_entity_name,
                                    new_entity_id=new_entity_name,
                                    index=key,
                                    additional_variables=variable_list)
        previous_entity_name = new_entity_name

    feature_matrix, feature_names = ft.dfs(entityset=entity_set,
                                           target_entity='entity_id_1',
                                           max_depth=2,
                                           verbose=1,
                                           n_jobs=3)

    feature_matrix = feature_matrix.reindex(index=primary_key)
    feature_matrix = feature_matrix.reset_index()

    return feature_matrix
