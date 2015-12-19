/*
 * Copyright 2015 recommenders.net.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.recommenders.rival.recommend.frameworks;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.grouplens.lenskit.scored.ScoredId;
import es.uam.eps.ir.ranksys.core.IdDouble;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import net.recommenders.rival.core.DataModel;

/**
 * Recommender-related IO operations.
 *
 * @author <a href="http://github.com/alansaid">Alan</a>.
 */
public final class RecommenderIO {

    /**
     * Utility classes should not have a public or default constructor.
     */
    private RecommenderIO() {
    }

    /**
     * Write recommendations to file.
     *
     * @param user the user
     * @param recommendations the recommendations
     * @param path directory where fileName will be written (if not null)
     * @param fileName name of the file, if null recommendations will not be
     * printed
     * @param append flag to decide if recommendations should be appended to
     * file
     * @param model if not null, recommendations will be saved here
     * @param <T> type of recommendations
     */
    public static <T> void writeData(final long user, final List<T> recommendations, final String path, final String fileName, final boolean append, final DataModel<Long, Long> model) {
        BufferedWriter out = null;
        try {
            File dir = null;
            if (path != null) {
                dir = new File(path);
                if (!dir.isDirectory()) {
                    if (!dir.mkdir() && (fileName != null)) {
                        System.out.println("Directory " + path + " could not be created");
                        return;
                    }
                }
            }
            if ((path != null) && (fileName != null)) {
                out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path + "/" + fileName, append), "UTF-8"));
            }
            for (Object ri : recommendations) {
                if (ri instanceof RecommendedItem) {
                    RecommendedItem recItem = (RecommendedItem) ri;
                    if (out != null) {
                        out.write(user + "\t" + recItem.getItemID() + "\t" + recItem.getValue() + "\n");
                    }
                    if (model != null) {
                        model.addPreference(user, recItem.getItemID(), 1.0 * recItem.getValue());
                    }
                }
                if (ri instanceof ScoredId) {
                    ScoredId recItem = (ScoredId) ri;
                    if (out != null) {
                        out.write(user + "\t" + recItem.getId() + "\t" + recItem.getScore() + "\n");
                    }
                    if (model != null) {
                        model.addPreference(user, recItem.getId(), recItem.getScore());
                    }
                }
                if (ri instanceof IdDouble) {
                    @SuppressWarnings("unchecked")
                    IdDouble<Long> recItem = (IdDouble<Long>) ri;
                    if (out != null) {
                        out.write(user + "\t" + recItem.id + "\t" + recItem.v + "\n");
                    }
                    if (model != null) {
                        model.addPreference(user, recItem.id, recItem.v);
                    }
                }
            }
            if (out != null) {
                out.flush();
                out.close();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
//            logger.error(e.getMessage());
        } finally {
            if (out != null) {
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
