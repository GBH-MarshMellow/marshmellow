package com.gbh.gbh_mm.user.model.request;

import lombok.Data;
import lombok.Getter;

@Getter
public class RequestCreateSalary {
    private long salary;
    private int date;
    private String account;
}
