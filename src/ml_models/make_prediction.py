


def make_prediction(file_path, df, test_ids, sub_df):
    """
    Args:
        file_path: file-name with base-path as "submission"
        df: array with shape (test_df.shape[0], cv_fold)
        test_ids: test_ids
        sub_df: submission data-frame
        
    Return:
        output a file with given name
        
    Example: 
    >>> make_prediction(file_path, predictions, ts_unique_ids, sub)
    """
    predictions = np.mean(df, axis=1)
    sub_df = pd.DataFrame({"ID_code":test_ids})
    sub_df["target"] = predictions
    sub_df.columns = sub.columns

    sub_df.to_csv('submission/stacking/{}.csv'.format(file_path), index=None)
    print("successfully saved")
#     print(sub_df.shape)
#     print(sub_df.sample(10))
