def add_converted_inc(pp_df):
    def process_inc(row):
        r_check = row["persinc"]
        val = None
        if "p.w." in r_check:
            r_check = r_check.replace("p.w.", "").replace(" ", "").replace("$", "")
            if "+" in r_check:
                r_check = r_check.replace("+", "")
            elif "-" in r_check:
                r_check = r_check.split("-")[0]
            else:
                raise ValueError(f"Dunno I never seen this lol {r_check}")
            val = int(r_check)
        elif "Zero" in r_check:
            val = 0
        elif "Negative" in r_check:
            val = -1
        elif "Missing" in r_check:
            val = -2
        else:
            raise ValueError(f"Dunno I never seen this lol {r_check}")
        return val

    pp_df["inc_dummy"] = pp_df.apply(process_inc, axis=1)
    return pp_df
