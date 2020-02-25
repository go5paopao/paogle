

def fast_concat(df1, df2):
    assert(len(df1) == len(df2))
    for col in [c for c in df2.columns if c not in df1.columns]:
        df1[col] = df2[col].values
    return df1


def fast_merge(df1, df2, on):
    if isinstance(on, str):
        tmp = df1[[on]].merge(df2, how="left", on=on)
    elif isinstance(on, list):
        tmp = df1[on].merge(df2, how="left", on=on)
    else:
        raise("on is not valid type :{}".format(on))
    for col in [col for col in df2.columns if col != on]:
        df1[col] = tmp[col].values
    return df1
